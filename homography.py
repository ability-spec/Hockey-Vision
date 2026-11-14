"""Map players from broadcast frames onto the 2D rink template using a per-frame homography.

Pipeline per frame:
    1. match_keypoints()   work out which rink landmark each dot / circle / line / goal detection is
    2. compute_homography() fit image pixels -> rink feet, dropping outlier matches
    3. HomographyTracker   smooth H over time and bridge frames with too few landmarks
    4. project_players()   map each player's skate position (and the puck) onto the rink

Run on an existing run_models.py output:
    python homography.py outputs/<clip>/detections.json videos/<clip>.mp4
This writes, next to the detections file:
    positions.json    per frame: the homography, camera view and cuts, keypoints used, and player /
                      puck rink positions
    positions.csv     one row per player or puck per frame: rink x, y in feet (and track id, jersey
                      number, and player name when run_models.py was given --teams)
    homographies.csv  one row per frame: the 3x3 image -> rink homography, h00..h22 (empty if no fit)
The video's frames are only decoded when they're needed: for the videos below, or to spot camera
cuts when detections.json is from a run_models.py that didn't record them. Frames run_models.py
skipped the rink models on (--rink-stride) keep the previous frame's fit. Videos are optional
outputs, for checking the fit by eye: --save-video writes side_by_side.mp4 (the video with the top-down rink
view beside it), and --debug adds radar.mp4 (top-down view with the keypoints the fit used) and
overlay.mp4 (rink keypoints reprojected onto the video).

A homography row maps a video pixel (u, v) to rink feet: [x, y, w] = H @ [u, v, 1], then
(x / w, y / w). Rink feet have the origin at the centre dot, x along the rink (-100 to 100) and
y across it (-42.5 near boards to 42.5 far boards); see rink.py.

In the videos, the top-down view shows players (with their jersey numbers, when run_models.py read them) and
the puck. On frames with no usable fit, including fits rejected because they put all the
players in a tiny patch or make them implausibly tall or short, the rink is shown empty with
its outline in red.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

import rink
from rink import KEYPOINTS
from video_io import CutDetector, FrameReader, video_info

MIN_POINTS = 4
OUTLIER_FT = 6.0  # a match further than this from the fit is a wrong identification
REFINE_ITERATIONS = 4  # rounds of re-matching circle box sides to the fitted circle
MIN_SPREAD_FT = 3.0  # matched points must spread at least this far off a single line

# How much each kind of match is trusted. Dots are small, so their box centre is precise;
# a circle's box centre is skewed by perspective; the goal box only roughly marks the goal line.
WEIGHT_DOT = 1.0
WEIGHT_LINE_END = 0.8
WEIGHT_CIRCLE = 0.6
WEIGHT_CIRCLE_EDGE = 0.5
WEIGHT_GOAL = 0.4

EDGE_MARGIN = 3  # px; a box side this close to the frame edge is cut off, not a real endpoint
# Players spread out over the ice; if they all project into a tiny patch, the homography is
# wrong (typically a zoomed-in close-up still using the fit from the wide shot before it).
# On real play the RMS distance of players from their centroid is almost always > 11 ft.
MIN_PLAYER_SPREAD_FT = 9.0
MIN_PLAYERS_FOR_SPREAD = 4
# A player's box height, converted to feet with the fit's scale at their skates, should come out
# at roughly a person's height (5-7 ft on real frames). Far outside that, the fit is wrong even
# if the players aren't clustered, e.g. a close-up where landmarks were misidentified.
PLAYER_HEIGHT_RANGE_FT = (3.0, 9.0)
MIN_PLAYERS_FOR_HEIGHT = 3
LINE_END_TOL_PX = 40  # a line box corner must be this close to where the fit predicts the line ends

PLAYER_CLASSES = {"team_a_player", "team_b_player", "goalie_a", "goalie_b", "referee"}
PLAYER_COLORS = {  # BGR
    "team_a_player": (60, 60, 220),
    "goalie_a": (0, 0, 140),
    "team_b_player": (220, 120, 30),
    "goalie_b": (140, 60, 0),
    "referee": (30, 30, 30),
}


class Box:
    def __init__(self, det):
        self.cls = det["class"]
        self.conf = det["confidence"]
        self.x1, self.y1, self.x2, self.y2 = det["box_xyxy"]
        self.jersey_number = det.get("jersey_number")
        self.player_name = det.get("player_name")  # with run_models.py --teams
        self.track_id = det.get("track_id")

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def cx(self):
        return self.center[0]

    @property
    def cy(self):
        return self.center[1]

    @property
    def bottom_center(self):
        return ((self.x1 + self.x2) / 2, self.y2)

    def contains(self, pt, pad=0.0):
        x, y = pt
        return self.x1 - pad <= x <= self.x2 + pad and self.y1 - pad <= y <= self.y2 + pad

    def spans_x(self, x):
        return self.x1 <= x <= self.x2


class Match:
    def __init__(self, keypoint, image_xy, weight, source, circle_edge=None):
        self.keypoint = keypoint
        self.image_xy = tuple(float(v) for v in image_xy)
        self.weight = weight
        self.source = source
        # Rink point this image point corresponds to. Starts at the keypoint; for a circle's box
        # side it is refined during fitting to the circle point that actually touches the box.
        self.rink_xy = KEYPOINTS[keypoint].xy
        # For circle box sides: (circle centre in feet, box side, box coordinate).
        self.circle_edge = circle_edge

    def copy(self):
        m = Match(self.keypoint, self.image_xy, self.weight, self.source, self.circle_edge)
        m.rink_xy = self.rink_xy
        return m

    def __repr__(self):
        return f"Match({self.keypoint}, {self.image_xy}, from={self.source})"


def _boxes(detections, model, cls=None):
    return [Box(d) for d in detections.get(model, []) if cls is None or d["class"] == cls]


def _dedupe(boxes, min_dist):
    """Drop boxes whose centre is within min_dist px of a more confident box."""
    kept = []
    for b in sorted(boxes, key=lambda b: -b.conf):
        if all(np.hypot(b.cx - k.cx, b.cy - k.cy) > min_dist for k in kept):
            kept.append(b)
    return kept


def _near_far_by_order(items, key_y):
    """For two items on the same end, the one lower in the image is on the near side."""
    items = sorted(items, key=key_y)
    return {id(items[0]): "far", id(items[-1]): "near"} if len(items) == 2 else {}


def _view_type(end_circles, goal):
    """"side" for the main broadcast camera (filmed from the near boards), "end" for a camera
    behind the blue line looking down the ice at the goal."""
    if goal is not None and end_circles:
        # Side view: the goal sits level with the circles (between the far and near one).
        # End view: the goal is at the far end, above every circle.
        return "end" if all(goal.y2 < c.y1 + 0.25 * (c.y2 - c.y1) for c in end_circles) else "side"
    if len(end_circles) == 2:
        a, b = sorted(end_circles, key=lambda c: c.cy)
        ha, hb = a.y2 - a.y1, b.y2 - b.y1
        # End view: both circles at a similar depth, so similar height and side by side.
        # Side view: the far circle is higher up and noticeably smaller.
        if abs(b.cy - a.cy) < 0.5 * (ha + hb) / 2 and min(ha, hb) / max(ha, hb) > 0.7:
            return "end"
    return "side"


def _circle_edge_matches(prefix, circle, view, frame_size, center=None):
    """Match the sides of a circle's box to the circle's extreme points.

    A circle seen in perspective is an ellipse, and each side of its box touches the ellipse at
    one point. As a starting guess that's the circle's matching left/right/top/bottom point;
    compute_homography() then refines which circle point it really is, since under a slanted
    view it can be several feet round the circle. Box sides cut off by the frame edge are skipped.
    `center` is the circle's dot when it was detected: more accurate than the box centre, and
    still right when the box is cut off.
    """
    w, h = frame_size
    cx, cy = center if center is not None else circle.center
    edges = {"left": (circle.x1, cy), "right": (circle.x2, cy), "top": (cx, circle.y1), "bottom": (cx, circle.y2)}
    clipped = {"left": circle.x1 <= EDGE_MARGIN, "right": circle.x2 >= w - EDGE_MARGIN,
               "top": circle.y1 <= EDGE_MARGIN, "bottom": circle.y2 >= h - EDGE_MARGIN}
    if view == "side":
        # Image right = +x, image up = +y: box sides line up with the keypoint names.
        names = {"left": "left", "right": "right", "top": "top", "bottom": "bottom"}
    else:
        # Looking at the right-end goal: image up = +x, image left = +y (far boards).
        names = {"top": "right", "bottom": "left", "left": "top", "right": "bottom"}
    if prefix == "center":  # centre circle keypoints are named far/near rather than top/bottom
        names = {k: {"top": "far", "bottom": "near"}.get(v, v) for k, v in names.items()}
    center_kp = "center_dot" if prefix == "center" else f"{prefix}_endzone_dot"
    box_value = {"left": circle.x1, "right": circle.x2, "top": circle.y1, "bottom": circle.y2}
    return [Match(f"{prefix}_circle_{names[side]}", pt, WEIGHT_CIRCLE_EDGE, "circle_edge",
                  (KEYPOINTS[center_kp].xy, side, box_value[side]))
            for side, pt in edges.items() if not clipped[side]]


class LineCandidate:
    """A blue line, goal line or centre line box whose corners may mark where the line meets
    the boards (or, for a goal line piece, the goal post).

    A straight line fills its box corner to corner along either the "/" or the "\\" diagonal,
    and the box alone doesn't say which. compute_homography() decides using the fit from the
    other landmarks: the diagonal whose corners land where that fit predicts the line ends.
    """

    def __init__(self, box, far_keypoint, near_keypoint, far_weight=WEIGHT_LINE_END,
                 near_weight=WEIGHT_LINE_END):
        self.box = box
        self.ends = {"far": (far_keypoint, far_weight), "near": (near_keypoint, near_weight)}
        self.view = "side"  # set by _line_candidates to the frame's camera view

    def corners(self, slash, view, frame_size):
        """{"far"/"near": image point} for one diagonal, leaving out corners cut off by the frame."""
        b = self.box
        a, c = ((b.x1, b.y2), (b.x2, b.y1)) if slash else ((b.x1, b.y1), (b.x2, b.y2))
        if view == "side":   # far boards are at the top of the image
            far, near = (a, c) if a[1] < c[1] else (c, a)
        else:                # far boards are on the left of the image
            far, near = (a, c) if a[0] < c[0] else (c, a)
        w, h = frame_size
        inside = lambda p: EDGE_MARGIN < p[0] < w - EDGE_MARGIN and EDGE_MARGIN < p[1] < h - EDGE_MARGIN
        return {side: pt for side, pt in (("far", far), ("near", near)) if inside(pt)}

    def matches(self, slash, view, frame_size, only=None):
        out = []
        for side, pt in self.corners(slash, view, frame_size).items():
            if only is None or side in only:
                name, weight = self.ends[side]
                out.append(Match(name, pt, weight, "line_end"))
        return out


def _line_candidates(detections, view, end, center_circle, goal):
    lines = []

    # Centre line: side view only.
    if view == "side":
        for cl in _dedupe(_boxes(detections, "rink", "Center_Line"), 100)[:1]:
            lines.append(LineCandidate(cl, "center_line_far_boards", "center_line_near_boards"))

    # Blue lines: which one is decided by what's next to it.
    blues = sorted(_dedupe(_boxes(detections, "rink", "Blue_Line"), 100), key=lambda b: b.cx)[:2]
    named = []
    if len(blues) == 2 and view == "side":
        named = [("L", blues[0]), ("R", blues[1])]
    elif len(blues) == 1:
        b = blues[0]
        if center_circle is not None and view == "side":
            named = [("L" if b.cx < center_circle.cx else "R", b)]
        elif end is not None:
            named = [(end, b)]  # the blue line next to the end zone in view
    for side, b in named:
        lines.append(LineCandidate(b, f"{side}_blue_line_far_boards", f"{side}_blue_line_near_boards"))

    # Goal line: usually detected as two pieces, split where the goal sits on it. The far piece
    # runs from the far boards to the far post, the near piece from the near post to the boards.
    if end is not None:
        pieces = _dedupe(_boxes(detections, "rink", "Goal_Back_Line"), 40)
        key = (lambda p: p.cy) if view == "side" else (lambda p: p.cx)
        pieces = sorted(pieces, key=key)
        if len(pieces) >= 2:
            labelled = [("far", pieces[0]), ("near", pieces[-1])]
        elif len(pieces) == 1 and goal is not None:
            labelled = [("far" if key(pieces[0]) < key(goal) else "near", pieces[0])]
        else:
            labelled = []
        for side, p in labelled:
            if side == "far":
                lines.append(LineCandidate(p, f"{end}_goal_line_far_boards", f"{end}_goal_post_far",
                                           near_weight=WEIGHT_GOAL))
            else:
                lines.append(LineCandidate(p, f"{end}_goal_post_near", f"{end}_goal_line_near_boards",
                                           far_weight=WEIGHT_GOAL))
    for line in lines:
        line.view = view
    return lines


def match_keypoints(detections, frame_size):
    """Assign rink keypoint names to the dot / circle / line / goal detections of one frame.

    `detections` is one frame's {model_name: [detection, ...]} dict as written by run_models.py,
    `frame_size` is (width, height). Returns (matches, lines, view): a list of Match, a list of
    LineCandidate whose corners compute_homography() resolves once it has a first fit, and the
    camera view ("side" or "end", below). A
    landmark whose identity can't be worked out is left out rather than guessed, because one
    wrong match can pull the whole homography off.

    Two camera angles are handled:
      side  the main broadcast camera, filmed from the near boards: left of the image is the
            left end of the rink (-x) and higher in the image is further away (+y).
      end   a camera behind the blue line looking at a goal. Which end that is can't be told
            from the ice markings alone (the rink is symmetric), so it's always treated as the
            right end (+x), with the far boards (+y) on the left of the image.
    """
    width, _ = frame_size
    dots = _dedupe(_boxes(detections, "dots"), 8)
    circles = _dedupe(_boxes(detections, "rink", "Circle"), 30)
    goals = _dedupe(_boxes(detections, "rink", "Goal_Posts") or _boxes(detections, "rink", "Goal_Zone"), 50)
    blue_lines = _boxes(detections, "rink", "Blue_Line")
    center_lines = _boxes(detections, "rink", "Center_Line")
    goal = max(goals, key=lambda g: g.conf) if goals else None

    # ---- Which circle is the centre circle? The centre line runs through it, and it sits
    # between the two blue lines. End circles sit next to a goal instead.
    center_circle = None
    if goal is None:
        for c in circles:
            on_center_line = any(cl.spans_x(c.cx) for cl in center_lines)
            between_blue = any(b.cx < c.cx for b in blue_lines) and any(b.cx > c.cx for b in blue_lines)
            if on_center_line or between_blue:
                center_circle = c
                break
    end_circles = [c for c in circles if c is not center_circle]
    if len(end_circles) > 2:  # keep the two biggest, the rest are usually partial duplicates
        end_circles = sorted(end_circles, key=lambda c: -(c.x2 - c.x1) * (c.y2 - c.y1))[:2]

    view = _view_type(end_circles, goal)

    # ---- Which end of the rink is in view?
    end = None  # "L" or "R"
    if view == "end":
        end = "R"
    elif goal is not None:
        others = [c.cx for c in end_circles] + [d.cx for d in dots]
        ref = np.mean(others) if others else width / 2
        end = "L" if goal.cx < ref else "R"
    elif end_circles and blue_lines:
        # End-zone circles are on the goal side of the blue line.
        end = "L" if np.mean([c.cx for c in end_circles]) < np.mean([b.cx for b in blue_lines]) else "R"
    elif end_circles and center_circle is not None:
        end = "L" if np.mean([c.cx for c in end_circles]) < center_circle.cx else "R"

    matches = []

    # ---- Centre circle and centre dot (side view only; an end camera doesn't see centre ice).
    if center_circle is not None and view == "side":
        inside = [d for d in dots if center_circle.contains(d.center)]
        if inside:
            d = min(inside, key=lambda d: np.hypot(d.cx - center_circle.cx, d.cy - center_circle.cy))
            matches.append(Match("center_dot", d.center, WEIGHT_DOT, "dot"))
            dots.remove(d)
            center_xy = d.center
        else:
            matches.append(Match("center_dot", center_circle.center, WEIGHT_CIRCLE, "circle"))
            center_xy = None
        matches.extend(_circle_edge_matches("center", center_circle, "side", frame_size, center_xy))

    # ---- End-zone circles and their dots.
    if end is not None and end_circles:
        if view == "side":
            order_key = lambda c: c.cy          # far circle is higher in the image
            single_side = (lambda c: "near" if c.cy > goal.cy else "far") if goal is not None else None
        else:
            order_key = lambda c: c.cx          # far circle is on the left
            single_side = (lambda c: "far" if c.cx < goal.cx else "near") if goal is not None else None
        sides = _near_far_by_order(end_circles, key_y=order_key)
        for c in end_circles:
            side = sides.get(id(c)) or (single_side(c) if single_side else None)
            if side is None:
                continue
            prefix = f"{end}_{side}"
            inside = [d for d in dots if c.contains(d.center)]
            if inside:
                d = min(inside, key=lambda d: np.hypot(d.cx - c.cx, d.cy - c.cy))
                matches.append(Match(f"{prefix}_endzone_dot", d.center, WEIGHT_DOT, "dot"))
                dots.remove(d)
                matches.extend(_circle_edge_matches(prefix, c, view, frame_size, d.center))
            else:
                matches.append(Match(f"{prefix}_endzone_dot", c.center, WEIGHT_CIRCLE, "circle"))
                matches.extend(_circle_edge_matches(prefix, c, view, frame_size))
    # Dots inside an end circle that couldn't be named aren't neutral-zone dots.
    for c in end_circles:
        dots = [d for d in dots if not c.contains(d.center)]

    # ---- Neutral-zone dots (side view): whatever dots are left, outside every circle.
    if view == "side":
        neutral = [d for d in dots if not any(c.contains(d.center, pad=10) for c in circles)]
        by_half = {"L": [], "R": []}
        for d in neutral:
            half = None
            if center_lines:
                half = "L" if d.cx < np.mean([cl.cx for cl in center_lines]) else "R"
            elif center_circle is not None:
                half = "L" if d.cx < center_circle.cx else "R"
            elif blue_lines:
                # Neutral dots are 5 ft inside the blue line, towards centre ice.
                nearest = min(blue_lines, key=lambda b: abs(b.cx - d.cx))
                half = "L" if nearest.cx < d.cx else "R"
            if half is not None:
                by_half[half].append(d)
        for half, ds in by_half.items():
            ds = sorted(ds, key=lambda d: d.cy)
            if len(ds) > 2:
                ds = [ds[0], ds[-1]]
            sides = _near_far_by_order(ds, key_y=lambda d: d.cy)
            for d in ds:
                side = sides.get(id(d))
                if side is None and center_circle is not None:
                    side = "near" if d.cy > center_circle.cy else "far"
                if side is not None:
                    matches.append(Match(f"{half}_neutral_dot_{side}", d.center, WEIGHT_DOT, "dot"))

    # ---- Lines: blue lines, goal lines and the centre line run board to board, so the two box
    # corners on the line's diagonal are where it meets the boards.
    lines = _line_candidates(detections, view, end, center_circle, goal)

    # ---- Goal: bottom-centre of the goal box is roughly where the goal line crosses y = 0.
    if goal is not None and end is not None and goal.cls == "Goal_Posts":
        matches.append(Match(f"{end}_goal_center", goal.bottom_center, WEIGHT_GOAL, "goal"))

    # One match per keypoint, keeping the most trusted.
    best = {}
    for m in matches:
        if m.keypoint not in best or m.weight > best[m.keypoint].weight:
            best[m.keypoint] = m
    return list(best.values()), lines, view


_CIRCLE_ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)


def _refine_circle_edges(matches, H):
    """Move each circle box-side match to the circle point that touches that box side under H."""
    H_inv = np.linalg.inv(H)
    for m in matches:
        if m.circle_edge is None:
            continue
        center, side, value = m.circle_edge
        ring = np.column_stack([center[0] + rink.CIRCLE_RADIUS * np.cos(_CIRCLE_ANGLES),
                                center[1] + rink.CIRCLE_RADIUS * np.sin(_CIRCLE_ANGLES)]).astype(np.float32)
        img = cv2.perspectiveTransform(ring.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        axis = 0 if side in ("left", "right") else 1
        i = int(np.argmin(img[:, axis]) if side in ("left", "top") else np.argmax(img[:, axis]))
        m.rink_xy = tuple(ring[i])
        m.image_xy = (value, float(img[i, 1])) if axis == 0 else (float(img[i, 0]), value)


def _fit(matches):
    img = np.float32([m.image_xy for m in matches])
    ft = np.float32([m.rink_xy for m in matches])
    H, _ = cv2.findHomography(img, ft, 0)
    # A singular fit is no fit: it can't be inverted (_refine_circle_edges, _resolve_lines would
    # raise and end the whole clip). The Rust port applies the same rule.
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        return None, None
    err = np.linalg.norm(cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2) - ft, axis=1)
    return H, err


def _robust_fit(matches, frame_size):
    """Least-squares fit over every match, refining circle box sides, then dropping the worst
    match and refitting while any is off by more than OUTLIER_FT. With only 5-10 rough points
    per frame this beats RANSAC, which tends to lock onto 4 points that happen to fit each other
    exactly. Returns (H, inlier_matches, errors) or (None, [], None)."""
    matches = [m.copy() for m in matches]
    while len(matches) >= MIN_POINTS:
        ft = np.float32([KEYPOINTS[m.keypoint].xy for m in matches])
        # Points (nearly) all on one line, e.g. only dots on one side, can't fix a homography:
        # require a few feet of spread in both directions.
        if np.linalg.svd(ft - ft.mean(axis=0), compute_uv=False)[1] / np.sqrt(len(ft)) < MIN_SPREAD_FT:
            return None, [], None
        H, err = _fit(matches)
        for _ in range(REFINE_ITERATIONS):
            if H is None:
                break
            _refine_circle_edges(matches, H)
            H, err = _fit(matches)
        if H is None:
            return None, [], None
        worst = int(np.argmax(err))
        if err[worst] <= OUTLIER_FT or len(matches) == MIN_POINTS:
            break
        matches.pop(worst)
    else:
        return None, [], None
    if err.max() > OUTLIER_FT or not _plausible(H, frame_size):
        return None, [], None
    return H, matches, err


def _resolve_lines(lines, H, frame_size):
    """Pick each line box's diagonal from where H predicts the line's ends, and return the
    corners that agree with the prediction as matches."""
    H_inv = np.linalg.inv(H)
    out = []
    for line in lines:
        names = {side: name for side, (name, _) in line.ends.items()}
        pred = {side: cv2.perspectiveTransform(np.float32([[KEYPOINTS[n].xy]]), H_inv)[0, 0]
                for side, n in names.items()}
        tol = max(LINE_END_TOL_PX, 0.2 * np.hypot(line.box.x2 - line.box.x1, line.box.y2 - line.box.y1))
        best = None
        for slash in (True, False):
            corners = line.corners(slash, line.view, frame_size)
            dists = {side: float(np.hypot(*(np.array(pt) - pred[side]))) for side, pt in corners.items()}
            good = {side for side, dist in dists.items() if dist < tol}
            if good and (best is None or len(good) > len(best[1]) or
                         (len(good) == len(best[1]) and sum(dists[s] for s in good) < best[2])):
                best = (slash, good, sum(dists[s] for s in good))
        if best is not None:
            out.extend(line.matches(best[0], line.view, frame_size, only=best[1]))
    return out


def compute_homography(matches, lines, frame_size):
    """Fit image -> rink (feet). Returns (H, inlier_matches) or (None, []).

    Fits the point landmarks first, then uses that fit to resolve which corners of each line
    box are the line's ends, and refits with them. When the points alone are too few for a
    first fit, every combination of line diagonals is tried and the most consistent fit kept.
    """
    H, inliers, _ = _robust_fit(matches, frame_size)
    if not lines:
        return (H, inliers) if H is not None else (None, [])

    if H is not None:
        extra = _resolve_lines(lines, H, frame_size)
        if extra:
            H2, inliers2, _ = _robust_fit(inliers + extra, frame_size)
            if H2 is not None:
                return H2, inliers2
        return H, inliers

    # No fit from points alone: try each diagonal per line. Needs more than MIN_POINTS matches,
    # since exactly MIN_POINTS always fit perfectly and can't tell a right guess from a wrong one.
    best = (None, [], np.inf)
    for combo in itertools.product((True, False), repeat=len(lines)):
        extra = [m for line, slash in zip(lines, combo) for m in line.matches(slash, line.view, frame_size)]
        if len(matches) + len(extra) <= MIN_POINTS:
            continue
        H, inliers, err = _robust_fit(matches + extra, frame_size)
        if H is not None and len(inliers) > MIN_POINTS and err.mean() < best[2]:
            best = (H, inliers, err.mean())
    return best[0], best[1]


def _plausible(H, frame_size):
    """Reject degenerate fits (mirrored, collapsed or absurdly scaled) before they reach the radar."""
    if abs(np.linalg.det(H)) < 1e-12:
        return False
    w, h = frame_size
    c = np.float32([[[w / 2, h / 2]], [[w / 2 + 100, h / 2]], [[w / 2, h / 2 - 100]]])
    p = cv2.perspectiveTransform(c, H)[:, 0]
    right, up = p[1] - p[0], p[2] - p[0]
    # Image y points down but rink y points to the far boards, so a correctly oriented
    # (unmirrored) view has image-right x image-up turning the same way as rink x x rink y.
    if right[0] * up[1] - right[1] * up[0] <= 0:
        return False
    # 100 px at the centre of a broadcast frame covers somewhere between ~0.5 and ~40 ft of ice.
    return all(0.5 < np.hypot(*v) < 40 for v in (right, up))


def image_to_rink(points_xy, H):
    pts = np.float32(points_xy).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def rink_to_image(points_ft, H):
    pts = np.float32(points_ft).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)


class HomographyTracker:
    """Smooths H across frames and reuses it for a while when a frame has too few landmarks."""

    def __init__(self, smoothing=0.5, max_age=10, max_jump_ft=8.0):
        self.smoothing = smoothing      # weight of the previous H; 0 = no smoothing
        self.max_age = max_age          # frames to keep reusing the last H before giving up
        self.max_jump_ft = max_jump_ft  # bigger frame-to-frame change than this = new shot
        self.H = None
        self.age = 0

    def reset(self):
        self.H, self.age = None, 0

    def _jump_ft(self, H_new, frame_size):
        """How far (ft) the old and new H disagree about where a grid of image points lands."""
        w, h = frame_size
        grid = np.float32([[x, y] for x in (0.2 * w, 0.5 * w, 0.8 * w) for y in (0.4 * h, 0.7 * h)])
        return float(np.max(np.linalg.norm(image_to_rink(grid, self.H) - image_to_rink(grid, H_new), axis=1)))

    def update(self, H_new, frame_size):
        if H_new is not None:
            H_new = H_new / H_new[2, 2]
            if self.H is None or self._jump_ft(H_new, frame_size) > self.max_jump_ft:
                # A camera cut or a big pan: start from the new fit instead of blending into it.
                self.H = H_new
            else:
                self.H = self.smoothing * self.H + (1 - self.smoothing) * H_new
            self.age = 0
        elif self.H is not None:
            self.age += 1
            if self.age > self.max_age:
                self.reset()
        return self.H


def project_players(detections, H):
    """Project each player's skate position (bottom-centre of the box) to rink feet."""
    players = [b for b in _boxes(detections, "player") if b.cls in PLAYER_CLASSES]
    if not players or H is None:
        return []
    ft = image_to_rink([p.bottom_center for p in players], H)
    out = []
    for p, (x, y) in zip(players, ft):
        # Anything well outside the boards is a bad projection (or a player on the bench).
        if abs(x) <= rink.HALF_LENGTH + 5 and abs(y) <= rink.HALF_WIDTH + 5:
            out.append({"class": p.cls, "confidence": p.conf, "jersey_number": p.jersey_number,
                        "player_name": p.player_name,
                        "track_id": p.track_id, "box_xyxy": [p.x1, p.y1, p.x2, p.y2],
                        "rink_xy": [round(float(np.clip(x, -rink.HALF_LENGTH, rink.HALF_LENGTH)), 2),
                                    round(float(np.clip(y, -rink.HALF_WIDTH, rink.HALF_WIDTH)), 2)]})
    return out


def players_clustered(players):
    """True when enough players are visible but they all land in an implausibly small patch."""
    if len(players) < MIN_PLAYERS_FOR_SPREAD:
        return False
    xy = np.array([p["rink_xy"] for p in players])
    spread = np.sqrt(np.mean(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))
    return spread < MIN_PLAYER_SPREAD_FT


def implied_player_height(detections, H):
    """Median player height in feet implied by H, or None with too few players.

    Uses the fit's horizontal scale at each player's skates: for an upright player, feet per
    pixel across the image at their feet is about the same as up their body.
    """
    boxes = [b for b in _boxes(detections, "player") if b.cls in PLAYER_CLASSES]
    if H is None or len(boxes) < MIN_PLAYERS_FOR_HEIGHT:
        return None
    heights = []
    for b in boxes:
        x, y = b.bottom_center
        left, right = image_to_rink([(x - 20, y), (x + 20, y)], H)
        heights.append((b.y2 - b.y1) * np.linalg.norm(right - left) / 40)
    return float(np.median(heights))


def project_puck(detections, H):
    """Project the most confident puck detection to rink feet, or None.

    The puck sits on the ice (and is tiny), so the centre of its box is its ice position. A puck
    in the air projects to where the camera ray meets the ice, so it can look further away than
    it is.
    """
    pucks = _boxes(detections, "puck", "puck")
    if not pucks or H is None:
        return None
    best = max(pucks, key=lambda b: b.conf)
    x, y = image_to_rink([best.center], H)[0]
    if abs(x) > rink.HALF_LENGTH or abs(y) > rink.HALF_WIDTH:
        return None
    return {"confidence": best.conf, "rink_xy": [round(float(x), 2), round(float(y), 2)]}


BOARDS_COLOR_NO_FIT = (40, 40, 220)  # BGR red: the rink outline when this frame has no usable fit
PUCK_COLOR = (0, 0, 0)
PUCK_RING_COLOR = (0, 220, 255)  # BGR yellow ring so the black puck stands out on the rink


def no_fit_template(template):
    """The template with the boards outline recoloured red."""
    img = template.astype(np.float32)
    # The boards are the only grey on the template (everything else is white, red or blue).
    # Blend each grey pixel towards red by how dark it is, so the anti-aliased edges follow too.
    grey = (img.max(axis=2) - img.min(axis=2) < 25) & (img.mean(axis=2) < 235)
    darkness = np.clip((250 - img.mean(axis=2)) / (250 - 64), 0, 1)[..., None]
    red = np.array(BOARDS_COLOR_NO_FIT, np.float32)
    white = np.array(PANEL_BG, np.float32)
    img[grey] = (darkness * red + (1 - darkness) * white)[grey]
    return img.astype(np.uint8)


# Referees are only detected so they aren't mistaken for players; they're left off the top-down view.
HIDDEN_ON_RADAR = {"referee"}


def draw_radar(template, players, puck=None, matches=()):
    """Draw players (and the puck) on the rink template, with each player's jersey number in their
    dot when it's known. Referees aren't drawn. `matches`, if given, also draws the keypoints the
    fit used, for debugging."""
    img = template.copy()
    if matches:
        img = rink.draw_keypoints(img, names={m.keypoint for m in matches}, labels=False)
    for p in players:
        if p["class"] in HIDDEN_ON_RADAR:
            continue
        x, y = map(int, np.round(rink.rink_to_template(p["rink_xy"])[0]))
        number = p.get("jersey_number")
        radius = 13 if number else 9  # room for the jersey number inside the dot
        cv2.circle(img, (x, y), radius + 2, (255, 255, 255), -1)
        cv2.circle(img, (x, y), radius, PLAYER_COLORS.get(p["class"], (0, 0, 0)), -1)
        if number:
            (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(img, number, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
    if puck is not None:
        x, y = map(int, np.round(rink.rink_to_template(puck["rink_xy"])[0]))
        cv2.circle(img, (x, y), 8, PUCK_RING_COLOR, -1)
        cv2.circle(img, (x, y), 5, PUCK_COLOR, -1)
    return img


# Template rows/columns that hold the rink (plus a small margin); the rest is blank padding.
RADAR_CROP = (slice(190, 790), slice(50, 1240))
PANEL_BG = (254, 253, 255)  # BGR, same off-white as the template background


def side_by_side(frame, radar, panel_width):
    """The untouched video frame on the left, the radar centred in a panel on the right."""
    h = frame.shape[0]
    rink_img = radar[RADAR_CROP]
    scale = min(panel_width / rink_img.shape[1], h / rink_img.shape[0])
    rink_img = cv2.resize(rink_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    panel = np.full((h, panel_width, 3), PANEL_BG, dtype=np.uint8)
    y = (h - rink_img.shape[0]) // 2
    x = (panel_width - rink_img.shape[1]) // 2
    panel[y:y + rink_img.shape[0], x:x + rink_img.shape[1]] = rink_img
    return np.hstack([frame, panel])


def draw_overlay(frame, H, matches):
    """Reproject every rink keypoint onto the frame; if H is right they sit on the markings."""
    img = frame.copy()
    if H is not None:
        names = list(KEYPOINTS)
        pts = rink_to_image([KEYPOINTS[n].xy for n in names], H)
        h, w = img.shape[:2]
        for n, (x, y) in zip(names, pts):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (int(x), int(y)), 6, rink.GROUP_COLORS[KEYPOINTS[n].group], -1)
    for m in matches:  # the detections used for the fit, as rings
        cv2.circle(img, tuple(int(v) for v in m.image_xy), 12, (0, 255, 255), 2)
        cv2.putText(img, m.keypoint, (int(m.image_xy[0]) + 14, int(m.image_xy[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def write_csvs(positions, out_dir):
    """Flat versions of positions.json, for spreadsheets and pandas."""
    with open(out_dir / "positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "object", "track_id", "class", "jersey_number", "player_name",
                    "confidence", "x_ft", "y_ft"])
        for p in positions:
            for pl in p["players"]:
                tid = pl.get("track_id")
                w.writerow([p["frame"], p["time_s"], "player", "" if tid is None else tid, pl["class"],
                            pl.get("jersey_number") or "", pl.get("player_name") or "", pl["confidence"],
                            *pl["rink_xy"]])
            if p["puck"] is not None:
                w.writerow([p["frame"], p["time_s"], "puck", "", "puck", "", "", p["puck"]["confidence"],
                            *p["puck"]["rink_xy"]])
    with open(out_dir / "homographies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "view", "rejected", "keypoints_used"] + [f"h{r}{c}" for r in range(3) for c in range(3)])
        for p in positions:
            H = sum(p["homography"], []) if p["homography"] is not None else [""] * 9
            w.writerow([p["frame"], p["time_s"], p["view"], p["rejected"] or "", " ".join(p["keypoints_used"])] + H)


def main():
    parser = argparse.ArgumentParser(
        description="Map detections onto the rink (feet) with a per-frame homography.")
    parser.add_argument("detections", type=Path, help="detections.json written by run_models.py for the video")
    parser.add_argument("video", type=Path, help="The source video the detections came from")
    parser.add_argument("--output", type=Path, default=None,
                        help="With --save-video, the output video (default: side_by_side.mp4 next to the detections)")
    parser.add_argument("--save-video", action="store_true",
                        help="Also write side_by_side.mp4, the video with the top-down rink view beside it")
    parser.add_argument("--debug", action="store_true",
                        help="Also write radar.mp4 and overlay.mp4 with the fitted keypoints drawn on")
    args = parser.parse_args()

    frames = json.loads(args.detections.read_text())
    if isinstance(frames, dict):  # single image output
        frames = [{"frame": 0, "time_s": 0.0, "detections": frames}]
    out_dir = args.detections.parent
    output = args.output or out_dir / "side_by_side.mp4"

    # run_models.py records camera cuts while it has the frames decoded; older detections.json
    # files don't have them, and then the video is decoded here to find them.
    need_frames = args.save_video or args.debug or any("cut" not in f for f in frames)
    try:
        fps, width, height, _ = video_info(args.video)
        video = FrameReader(args.video, len(frames)) if need_frames else None
    except OSError as e:
        raise SystemExit(str(e))
    frame_size = (width, height)
    panel_width = int(width * 0.75)

    template = rink.load_template()
    template_no_fit = no_fit_template(template)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    if args.save_video:
        out = cv2.VideoWriter(str(output), fourcc, fps, (width + panel_width, height))
    radar_out = overlay_out = None
    if args.debug:
        radar_out = cv2.VideoWriter(str(out_dir / "radar.mp4"), fourcc, fps, template.shape[1::-1])
        overlay_out = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps, frame_size)

    tracker = HomographyTracker()
    positions = []
    rejected_counts = {"players_clustered": 0, "player_scale": 0}
    cuts = CutDetector()
    view = "side"
    try:
        for f, frame in zip(frames, video if video is not None else itertools.repeat(None)):
            dets = f["detections"]

            cut = f["cut"] if "cut" in f else cuts.update(frame)
            if cut:
                tracker.reset()

            if f.get("rink_skipped"):
                # run_models.py --rink-stride didn't run the rink models: keep the last fit.
                inliers, H = [], tracker.H
            else:
                matches, lines, view = match_keypoints(dets, frame_size)
                H_frame, inliers = compute_homography(matches, lines, frame_size)
                H = tracker.update(H_frame, frame_size)
            players = project_players(dets, H)
            puck = project_puck(dets, H)

            rejected = None
            player_height = implied_player_height(dets, H)
            if players_clustered(players):
                rejected = "players_clustered"
            elif player_height is not None and not (
                    PLAYER_HEIGHT_RANGE_FT[0] <= player_height <= PLAYER_HEIGHT_RANGE_FT[1]):
                rejected = "player_scale"
            if rejected:
                # The fit is wrong: drop this frame's output and don't let the tracker carry
                # the bad fit forward.
                H, players, puck, inliers = None, [], None, []
                tracker.reset()
                rejected_counts[rejected] += 1

            positions.append({"frame": f["frame"], "time_s": f["time_s"],
                              "homography": None if H is None else np.round(H, 8).tolist(),
                              "cut": bool(cut),
                              "view": view,
                              "rejected": rejected,
                              "keypoints_used": [m.keypoint for m in inliers],
                              "players": players,
                              "puck": puck})

            # The rink outline turns red on frames with no usable fit (too few landmarks, or rejected).
            base = template if H is not None else template_no_fit
            if out is not None:
                out.write(side_by_side(frame, draw_radar(base, players, puck), panel_width))
            if args.debug:
                radar_out.write(draw_radar(base, players, puck, inliers))
                overlay_out.write(draw_overlay(frame, H, inliers))
            if len(positions) % 50 == 0 or len(positions) == len(frames):
                print(f"\r  frame {len(positions)}/{len(frames)}", end="", flush=True)
    finally:
        print()
        if video is not None:
            video.close()
        for w in (out, radar_out, overlay_out):
            if w is not None:
                w.release()

    (out_dir / "positions.json").write_text(json.dumps(positions))
    write_csvs(positions, out_dir)
    shown = sum(p["homography"] is not None for p in positions)
    print(f"Rink fit on {shown}/{len(frames)} frames (of the other {len(positions) - shown}: "
          f"{rejected_counts['players_clustered']} with clustered players, "
          f"{rejected_counts['player_scale']} with implausible player size, rest too few rink markings); "
          f"puck on the rink in {sum(p['puck'] is not None for p in positions)}")
    written = ([output] if args.save_video else []) + [out_dir / n for n in
                                                        ("positions.json", "positions.csv", "homographies.csv")]
    if args.debug:
        written += [out_dir / "radar.mp4", out_dir / "overlay.mp4"]
    print("Wrote " + ", ".join(str(w) for w in written))


if __name__ == "__main__":
    main()
