//! Port of `homography.py`'s per-frame work: name the rink landmarks a frame's detections show,
//! fit image pixels -> rink feet, smooth the fit over time, and project players and the puck.
//!
//! The functions mirror the Python ones (same names without the leading underscore, same order
//! of operations, same thresholds), so a change to one is easy to carry over to the other.
//! Python's float32 arrays are reproduced by rounding to `f32` where numpy would ([`f32r`]).
//!
//! One deliberate difference: a fit whose matrix is singular counts as no fit ([`fit`]). In Python
//! `np.linalg.inv` raises on it and ends the run (see `rust/bench/run_python_homography.py`).

use std::sync::OnceLock;

use crate::detections::{Detection, FrameRecord};
use crate::geometry::{f32r, f32r2, find_homography, invert, transform, Mat3};
use crate::rink::{self, keypoints, KeypointId};

pub const MIN_POINTS: usize = 4;
pub const OUTLIER_FT: f64 = 6.0;
pub const REFINE_ITERATIONS: usize = 4;
pub const MIN_SPREAD_FT: f64 = 3.0;

pub const WEIGHT_DOT: f64 = 1.0;
pub const WEIGHT_LINE_END: f64 = 0.8;
pub const WEIGHT_CIRCLE: f64 = 0.6;
pub const WEIGHT_CIRCLE_EDGE: f64 = 0.5;
pub const WEIGHT_GOAL: f64 = 0.4;

pub const EDGE_MARGIN: f64 = 3.0;
pub const MIN_PLAYER_SPREAD_FT: f64 = 9.0;
pub const MIN_PLAYERS_FOR_SPREAD: usize = 4;
pub const PLAYER_HEIGHT_RANGE_FT: (f64, f64) = (3.0, 9.0);
pub const MIN_PLAYERS_FOR_HEIGHT: usize = 3;
pub const LINE_END_TOL_PX: f64 = 40.0;

pub const PLAYER_CLASSES: [&str; 5] = [
    "team_a_player",
    "team_b_player",
    "goalie_a",
    "goalie_b",
    "referee",
];

/// (width, height) of the video frame, px.
pub type FrameSize = (f64, f64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Side,
    End,
}

impl View {
    pub fn as_str(self) -> &'static str {
        match self {
            View::Side => "side",
            View::End => "end",
        }
    }
}

/// A detection box (homography.py's `Box`).
#[derive(Clone, Copy, Debug)]
pub struct Bx<'a> {
    pub det: &'a Detection,
    pub conf: f64,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl<'a> Bx<'a> {
    fn new(det: &'a Detection) -> Self {
        let [x1, y1, x2, y2] = det.box_xyxy;
        Bx {
            det,
            conf: det.confidence,
            x1,
            y1,
            x2,
            y2,
        }
    }
    pub fn center(&self) -> [f64; 2] {
        [(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0]
    }
    pub fn cx(&self) -> f64 {
        (self.x1 + self.x2) / 2.0
    }
    pub fn cy(&self) -> f64 {
        (self.y1 + self.y2) / 2.0
    }
    pub fn bottom_center(&self) -> [f64; 2] {
        [(self.x1 + self.x2) / 2.0, self.y2]
    }
    pub fn contains(&self, p: [f64; 2], pad: f64) -> bool {
        self.x1 - pad <= p[0]
            && p[0] <= self.x2 + pad
            && self.y1 - pad <= p[1]
            && p[1] <= self.y2 + pad
    }
    pub fn spans_x(&self, x: f64) -> bool {
        self.x1 <= x && x <= self.x2
    }
    fn same(&self, other: &Bx) -> bool {
        std::ptr::eq(self.det, other.det)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxSide {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Dot,
    Circle,
    CircleEdge,
    LineEnd,
    Goal,
}

/// A circle box side: (circle centre in feet, box side, box coordinate).
#[derive(Clone, Copy, Debug)]
pub struct CircleEdge {
    pub center: [f64; 2],
    pub side: BoxSide,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct Match {
    pub keypoint: KeypointId,
    pub image_xy: [f64; 2],
    pub weight: f64,
    pub source: Source,
    /// Rink point this image point corresponds to; refined for circle box sides.
    pub rink_xy: [f64; 2],
    pub circle_edge: Option<CircleEdge>,
}

impl Match {
    fn new(name: &str, image_xy: [f64; 2], weight: f64, source: Source) -> Self {
        let kp = keypoints().id(name);
        Match {
            keypoint: kp,
            image_xy,
            weight,
            source,
            rink_xy: keypoints().xy(kp),
            circle_edge: None,
        }
    }
}

fn boxes<'a>(frame: &'a FrameRecord, model: &str, cls: Option<&str>) -> Vec<Bx<'a>> {
    frame
        .model(model)
        .iter()
        .filter(|d| cls.map_or(true, |c| d.cls == c))
        .map(Bx::new)
        .collect()
}

/// Drop boxes whose centre is within min_dist px of a more confident box.
fn dedupe(mut bs: Vec<Bx>, min_dist: f64) -> Vec<Bx> {
    bs.sort_by(|a, b| b.conf.total_cmp(&a.conf));
    let mut kept: Vec<Bx> = Vec::new();
    for b in bs {
        if kept
            .iter()
            .all(|k| (b.cx() - k.cx()).hypot(b.cy() - k.cy()) > min_dist)
        {
            kept.push(b);
        }
    }
    kept
}

fn mean(v: impl Iterator<Item = f64>) -> Option<f64> {
    let (s, n) = v.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    (n > 0).then(|| s / n as f64)
}

/// For two items on the same end, the one first by `key` is "far" and the last "near".
fn near_far_by_order<'a>(
    items: &[Bx<'a>],
    key: impl Fn(&Bx) -> f64,
) -> Vec<(Bx<'a>, &'static str)> {
    if items.len() != 2 {
        return vec![];
    }
    let mut sorted = items.to_vec();
    sorted.sort_by(|a, b| key(a).total_cmp(&key(b)));
    vec![(sorted[0], "far"), (sorted[1], "near")]
}

fn side_of(sides: &[(Bx, &'static str)], b: &Bx) -> Option<&'static str> {
    sides.iter().find(|(s, _)| s.same(b)).map(|(_, n)| *n)
}

fn view_type(end_circles: &[Bx], goal: Option<&Bx>) -> View {
    if let (Some(goal), false) = (goal, end_circles.is_empty()) {
        return if end_circles
            .iter()
            .all(|c| goal.y2 < c.y1 + 0.25 * (c.y2 - c.y1))
        {
            View::End
        } else {
            View::Side
        };
    }
    if end_circles.len() == 2 {
        let mut s = end_circles.to_vec();
        s.sort_by(|a, b| a.cy().total_cmp(&b.cy()));
        let (a, b) = (s[0], s[1]);
        let (ha, hb) = (a.y2 - a.y1, b.y2 - b.y1);
        if (b.cy() - a.cy()).abs() < 0.5 * (ha + hb) / 2.0 && ha.min(hb) / ha.max(hb) > 0.7 {
            return View::End;
        }
    }
    View::Side
}

fn circle_edge_matches(
    prefix: &str,
    circle: &Bx,
    view: View,
    fs: FrameSize,
    center: Option<[f64; 2]>,
) -> Vec<Match> {
    let (w, h) = fs;
    let [cx, cy] = center.unwrap_or_else(|| circle.center());
    let edges = [
        (
            BoxSide::Left,
            [circle.x1, cy],
            circle.x1 <= EDGE_MARGIN,
            circle.x1,
        ),
        (
            BoxSide::Right,
            [circle.x2, cy],
            circle.x2 >= w - EDGE_MARGIN,
            circle.x2,
        ),
        (
            BoxSide::Top,
            [cx, circle.y1],
            circle.y1 <= EDGE_MARGIN,
            circle.y1,
        ),
        (
            BoxSide::Bottom,
            [cx, circle.y2],
            circle.y2 >= h - EDGE_MARGIN,
            circle.y2,
        ),
    ];
    let name = |side: BoxSide| -> &'static str {
        let n = match (view, side) {
            (View::Side, BoxSide::Left) => "left",
            (View::Side, BoxSide::Right) => "right",
            (View::Side, BoxSide::Top) => "top",
            (View::Side, BoxSide::Bottom) => "bottom",
            // Looking at the right-end goal: image up = +x, image left = +y (far boards).
            (View::End, BoxSide::Top) => "right",
            (View::End, BoxSide::Bottom) => "left",
            (View::End, BoxSide::Left) => "top",
            (View::End, BoxSide::Right) => "bottom",
        };
        if prefix == "center" {
            match n {
                "top" => "far",
                "bottom" => "near",
                other => other,
            }
        } else {
            n
        }
    };
    let center_kp = if prefix == "center" {
        "center_dot".to_string()
    } else {
        format!("{prefix}_endzone_dot")
    };
    let center_ft = keypoints().xy(keypoints().id(&center_kp));
    edges
        .iter()
        .filter(|(_, _, clipped, _)| !clipped)
        .map(|&(side, pt, _, value)| {
            let mut m = Match::new(
                &format!("{prefix}_circle_{}", name(side)),
                pt,
                WEIGHT_CIRCLE_EDGE,
                Source::CircleEdge,
            );
            m.circle_edge = Some(CircleEdge {
                center: center_ft,
                side,
                value,
            });
            m
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Far,
    Near,
}

/// A blue / goal / centre line box whose diagonal corners may be where the line meets the boards.
#[derive(Clone, Debug)]
pub struct LineCandidate {
    pub bx: [f64; 4],
    /// (keypoint, weight) for the far and the near end.
    pub far: (KeypointId, f64),
    pub near: (KeypointId, f64),
    pub view: View,
}

impl LineCandidate {
    fn new(b: &Bx, far: &str, near: &str, far_weight: f64, near_weight: f64) -> Self {
        let k = keypoints();
        LineCandidate {
            bx: [b.x1, b.y1, b.x2, b.y2],
            far: (k.id(far), far_weight),
            near: (k.id(near), near_weight),
            view: View::Side,
        }
    }

    /// Far / near image points for one diagonal, leaving out corners cut off by the frame.
    pub fn corners(&self, slash: bool, view: View, fs: FrameSize) -> Vec<(End, [f64; 2])> {
        let [x1, y1, x2, y2] = self.bx;
        let (a, c) = if slash {
            ([x1, y2], [x2, y1])
        } else {
            ([x1, y1], [x2, y2])
        };
        let (far, near) = match view {
            View::Side => {
                if a[1] < c[1] {
                    (a, c)
                } else {
                    (c, a)
                }
            }
            View::End => {
                if a[0] < c[0] {
                    (a, c)
                } else {
                    (c, a)
                }
            }
        };
        let (w, h) = fs;
        let inside = |p: [f64; 2]| {
            EDGE_MARGIN < p[0]
                && p[0] < w - EDGE_MARGIN
                && EDGE_MARGIN < p[1]
                && p[1] < h - EDGE_MARGIN
        };
        [(End::Far, far), (End::Near, near)]
            .into_iter()
            .filter(|(_, p)| inside(*p))
            .collect()
    }

    pub fn matches(
        &self,
        slash: bool,
        view: View,
        fs: FrameSize,
        only: Option<&[End]>,
    ) -> Vec<Match> {
        let k = keypoints();
        self.corners(slash, view, fs)
            .into_iter()
            .filter(|(side, _)| only.map_or(true, |o| o.contains(side)))
            .map(|(side, pt)| {
                let (kp, weight) = if side == End::Far {
                    self.far
                } else {
                    self.near
                };
                Match::new(k.name(kp), pt, weight, Source::LineEnd)
            })
            .collect()
    }

    fn end(&self, side: End) -> KeypointId {
        if side == End::Far {
            self.far.0
        } else {
            self.near.0
        }
    }
}

fn line_candidates(
    frame: &FrameRecord,
    view: View,
    end: Option<&str>,
    center_circle: Option<&Bx>,
    goal: Option<&Bx>,
) -> Vec<LineCandidate> {
    let mut lines = Vec::new();
    if view == View::Side {
        for cl in dedupe(boxes(frame, "rink", Some("Center_Line")), 100.0)
            .iter()
            .take(1)
        {
            lines.push(LineCandidate::new(
                cl,
                "center_line_far_boards",
                "center_line_near_boards",
                WEIGHT_LINE_END,
                WEIGHT_LINE_END,
            ));
        }
    }

    let mut blues = dedupe(boxes(frame, "rink", Some("Blue_Line")), 100.0);
    blues.sort_by(|a, b| a.cx().total_cmp(&b.cx()));
    blues.truncate(2);
    let mut named: Vec<(String, Bx)> = vec![];
    if blues.len() == 2 && view == View::Side {
        named = vec![("L".into(), blues[0]), ("R".into(), blues[1])];
    } else if blues.len() == 1 {
        let b = blues[0];
        if let (Some(cc), View::Side) = (center_circle, view) {
            named = vec![(if b.cx() < cc.cx() { "L" } else { "R" }.into(), b)];
        } else if let Some(e) = end {
            named = vec![(e.into(), b)];
        }
    }
    for (side, b) in &named {
        lines.push(LineCandidate::new(
            b,
            &format!("{side}_blue_line_far_boards"),
            &format!("{side}_blue_line_near_boards"),
            WEIGHT_LINE_END,
            WEIGHT_LINE_END,
        ));
    }

    if let Some(end) = end {
        let mut pieces = dedupe(boxes(frame, "rink", Some("Goal_Back_Line")), 40.0);
        let key = |p: &Bx| if view == View::Side { p.cy() } else { p.cx() };
        pieces.sort_by(|a, b| key(a).total_cmp(&key(b)));
        let labelled: Vec<(End, Bx)> = if pieces.len() >= 2 {
            vec![(End::Far, pieces[0]), (End::Near, *pieces.last().unwrap())]
        } else if let (1, Some(goal)) = (pieces.len(), goal) {
            vec![(
                if key(&pieces[0]) < key(goal) {
                    End::Far
                } else {
                    End::Near
                },
                pieces[0],
            )]
        } else {
            vec![]
        };
        for (side, p) in labelled {
            lines.push(match side {
                End::Far => LineCandidate::new(
                    &p,
                    &format!("{end}_goal_line_far_boards"),
                    &format!("{end}_goal_post_far"),
                    WEIGHT_LINE_END,
                    WEIGHT_GOAL,
                ),
                End::Near => LineCandidate::new(
                    &p,
                    &format!("{end}_goal_post_near"),
                    &format!("{end}_goal_line_near_boards"),
                    WEIGHT_GOAL,
                    WEIGHT_LINE_END,
                ),
            });
        }
    }
    for l in &mut lines {
        l.view = view;
    }
    lines
}

fn nearest_index(ds: &[Bx], c: [f64; 2]) -> usize {
    let mut best = 0;
    for (i, d) in ds.iter().enumerate() {
        if (d.cx() - c[0]).hypot(d.cy() - c[1]) < (ds[best].cx() - c[0]).hypot(ds[best].cy() - c[1])
        {
            best = i;
        }
    }
    best
}

/// Name the rink landmarks in one frame's detections. Returns (matches, line candidates, view);
/// see `match_keypoints` in homography.py.
pub fn match_keypoints(
    frame: &FrameRecord,
    fs: FrameSize,
) -> (Vec<Match>, Vec<LineCandidate>, View) {
    let (width, _) = fs;
    let mut dots = dedupe(boxes(frame, "dots", None), 8.0);
    let circles = dedupe(boxes(frame, "rink", Some("Circle")), 30.0);
    let posts = boxes(frame, "rink", Some("Goal_Posts"));
    let goals = dedupe(
        if posts.is_empty() {
            boxes(frame, "rink", Some("Goal_Zone"))
        } else {
            posts
        },
        50.0,
    );
    let blue_lines = boxes(frame, "rink", Some("Blue_Line"));
    let center_lines = boxes(frame, "rink", Some("Center_Line"));
    let goal = goals.first().copied(); // dedupe sorts by confidence, so this is the most confident

    let mut center_circle: Option<Bx> = None;
    if goal.is_none() {
        for c in &circles {
            let on_center_line = center_lines.iter().any(|cl| cl.spans_x(c.cx()));
            let between_blue = blue_lines.iter().any(|b| b.cx() < c.cx())
                && blue_lines.iter().any(|b| b.cx() > c.cx());
            if on_center_line || between_blue {
                center_circle = Some(*c);
                break;
            }
        }
    }
    let mut end_circles: Vec<Bx> = circles
        .iter()
        .filter(|c| center_circle.map_or(true, |cc| !cc.same(c)))
        .copied()
        .collect();
    if end_circles.len() > 2 {
        end_circles.sort_by(|a, b| {
            ((b.x2 - b.x1) * (b.y2 - b.y1)).total_cmp(&((a.x2 - a.x1) * (a.y2 - a.y1)))
        });
        end_circles.truncate(2);
    }

    let view = view_type(&end_circles, goal.as_ref());

    let end: Option<&str> = if view == View::End {
        Some("R")
    } else if let Some(g) = &goal {
        let r = mean(
            end_circles
                .iter()
                .map(|c| c.cx())
                .chain(dots.iter().map(|d| d.cx())),
        )
        .unwrap_or(width / 2.0);
        Some(if g.cx() < r { "L" } else { "R" })
    } else if !end_circles.is_empty() && !blue_lines.is_empty() {
        let ec = mean(end_circles.iter().map(|c| c.cx())).unwrap();
        let bl = mean(blue_lines.iter().map(|b| b.cx())).unwrap();
        Some(if ec < bl { "L" } else { "R" })
    } else if let (false, Some(cc)) = (end_circles.is_empty(), &center_circle) {
        Some(
            if mean(end_circles.iter().map(|c| c.cx())).unwrap() < cc.cx() {
                "L"
            } else {
                "R"
            },
        )
    } else {
        None
    };

    let mut matches: Vec<Match> = Vec::new();

    if let (Some(cc), View::Side) = (&center_circle, view) {
        let inside: Vec<Bx> = dots
            .iter()
            .filter(|d| cc.contains(d.center(), 0.0))
            .copied()
            .collect();
        let center_xy = if !inside.is_empty() {
            let d = inside[nearest_index(&inside, cc.center())];
            matches.push(Match::new(
                "center_dot",
                d.center(),
                WEIGHT_DOT,
                Source::Dot,
            ));
            dots.retain(|x| !x.same(&d));
            Some(d.center())
        } else {
            matches.push(Match::new(
                "center_dot",
                cc.center(),
                WEIGHT_CIRCLE,
                Source::Circle,
            ));
            None
        };
        matches.extend(circle_edge_matches("center", cc, View::Side, fs, center_xy));
    }

    if let (Some(end), false) = (end, end_circles.is_empty()) {
        let order_key = |c: &Bx| if view == View::Side { c.cy() } else { c.cx() };
        let sides = near_far_by_order(&end_circles, order_key);
        for c in &end_circles {
            let side = side_of(&sides, c).or_else(|| {
                goal.map(|g| match view {
                    View::Side => {
                        if c.cy() > g.cy() {
                            "near"
                        } else {
                            "far"
                        }
                    }
                    View::End => {
                        if c.cx() < g.cx() {
                            "far"
                        } else {
                            "near"
                        }
                    }
                })
            });
            let Some(side) = side else { continue };
            let prefix = format!("{end}_{side}");
            let inside: Vec<Bx> = dots
                .iter()
                .filter(|d| c.contains(d.center(), 0.0))
                .copied()
                .collect();
            if !inside.is_empty() {
                let d = inside[nearest_index(&inside, c.center())];
                matches.push(Match::new(
                    &format!("{prefix}_endzone_dot"),
                    d.center(),
                    WEIGHT_DOT,
                    Source::Dot,
                ));
                dots.retain(|x| !x.same(&d));
                matches.extend(circle_edge_matches(&prefix, c, view, fs, Some(d.center())));
            } else {
                matches.push(Match::new(
                    &format!("{prefix}_endzone_dot"),
                    c.center(),
                    WEIGHT_CIRCLE,
                    Source::Circle,
                ));
                matches.extend(circle_edge_matches(&prefix, c, view, fs, None));
            }
        }
    }
    // Dots inside an end circle that couldn't be named aren't neutral-zone dots.
    for c in &end_circles {
        dots.retain(|d| !c.contains(d.center(), 0.0));
    }

    if view == View::Side {
        let neutral: Vec<Bx> = dots
            .iter()
            .filter(|d| !circles.iter().any(|c| c.contains(d.center(), 10.0)))
            .copied()
            .collect();
        let mut by_half: [(&str, Vec<Bx>); 2] = [("L", vec![]), ("R", vec![])];
        let center_line_x = mean(center_lines.iter().map(|cl| cl.cx()));
        for d in neutral {
            let half = if let Some(x) = center_line_x {
                Some(if d.cx() < x { 0 } else { 1 })
            } else if let Some(cc) = &center_circle {
                Some(if d.cx() < cc.cx() { 0 } else { 1 })
            } else if !blue_lines.is_empty() {
                let mut nearest = &blue_lines[0];
                for b in &blue_lines {
                    if (b.cx() - d.cx()).abs() < (nearest.cx() - d.cx()).abs() {
                        nearest = b;
                    }
                }
                Some(if nearest.cx() < d.cx() { 0 } else { 1 })
            } else {
                None
            };
            if let Some(h) = half {
                by_half[h].1.push(d);
            }
        }
        for (half, ds) in by_half.iter_mut() {
            ds.sort_by(|a, b| a.cy().total_cmp(&b.cy()));
            if ds.len() > 2 {
                *ds = vec![ds[0], *ds.last().unwrap()];
            }
            let sides = near_far_by_order(ds, |d| d.cy());
            for d in ds.iter() {
                let mut side = side_of(&sides, d);
                if side.is_none() {
                    if let Some(cc) = &center_circle {
                        side = Some(if d.cy() > cc.cy() { "near" } else { "far" });
                    }
                }
                if let Some(side) = side {
                    matches.push(Match::new(
                        &format!("{half}_neutral_dot_{side}"),
                        d.center(),
                        WEIGHT_DOT,
                        Source::Dot,
                    ));
                }
            }
        }
    }

    let lines = line_candidates(frame, view, end, center_circle.as_ref(), goal.as_ref());

    if let (Some(g), Some(end)) = (&goal, end) {
        if g.det.cls == "Goal_Posts" {
            matches.push(Match::new(
                &format!("{end}_goal_center"),
                g.bottom_center(),
                WEIGHT_GOAL,
                Source::Goal,
            ));
        }
    }

    // One match per keypoint, keeping the most trusted (in order of first appearance, like a dict).
    let mut best: Vec<Match> = Vec::with_capacity(matches.len());
    for m in matches {
        match best.iter_mut().find(|b| b.keypoint == m.keypoint) {
            Some(b) => {
                if m.weight > b.weight {
                    *b = m;
                }
            }
            None => best.push(m),
        }
    }
    (best, lines, view)
}

fn ring_unit() -> &'static [[f64; 2]; 180] {
    static RING: OnceLock<[[f64; 2]; 180]> = OnceLock::new();
    RING.get_or_init(|| {
        let step = 2.0 * std::f64::consts::PI / 180.0;
        std::array::from_fn(|i| {
            let a = i as f64 * step;
            [a.cos(), a.sin()]
        })
    })
}

/// Move each circle box-side match to the circle point that touches that box side under `h`.
fn refine_circle_edges(matches: &mut [Match], h: &Mat3) {
    let Some(h_inv) = invert(h) else { return };
    for m in matches.iter_mut() {
        let Some(edge) = m.circle_edge else { continue };
        let mut best: Option<(usize, [f64; 2], [f64; 2])> = None; // (index, ring point, image point)
        for (i, u) in ring_unit().iter().enumerate() {
            let ring = [
                f32r(edge.center[0] + rink::CIRCLE_RADIUS * u[0]),
                f32r(edge.center[1] + rink::CIRCLE_RADIUS * u[1]),
            ];
            let img = transform(&h_inv, ring);
            let better = match (&best, edge.side) {
                (None, _) => true,
                (Some((_, _, b)), BoxSide::Left) => img[0] < b[0],
                (Some((_, _, b)), BoxSide::Right) => img[0] > b[0],
                (Some((_, _, b)), BoxSide::Top) => img[1] < b[1],
                (Some((_, _, b)), BoxSide::Bottom) => img[1] > b[1],
            };
            if better {
                best = Some((i, ring, img));
            }
        }
        let (_, ring, img) = best.unwrap();
        m.rink_xy = ring;
        m.image_xy = match edge.side {
            BoxSide::Left | BoxSide::Right => [edge.value, img[1]],
            BoxSide::Top | BoxSide::Bottom => [img[0], edge.value],
        };
    }
}

/// Least-squares fit of every match; (H, error of each match in ft). A singular H is no fit.
fn fit(matches: &[Match]) -> Option<(Mat3, Vec<f64>)> {
    let img: Vec<[f64; 2]> = matches.iter().map(|m| f32r2(m.image_xy)).collect();
    let ft: Vec<[f64; 2]> = matches.iter().map(|m| f32r2(m.rink_xy)).collect();
    let h = find_homography(&img, &ft)?;
    if h.determinant().abs() < 1e-12 {
        return None;
    }
    let err = img
        .iter()
        .zip(&ft)
        .map(|(p, f)| {
            let q = transform(&h, *p);
            f32r((f32r(q[0] - f[0])).hypot(f32r(q[1] - f[1])))
        })
        .collect();
    Some((h, err))
}

/// Second singular value of the centred keypoints / sqrt(n): their spread off a single line.
fn spread_off_line(matches: &[Match]) -> f64 {
    let k = keypoints();
    let pts: Vec<[f64; 2]> = matches.iter().map(|m| f32r2(k.xy(m.keypoint))).collect();
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
    for p in &pts {
        let (dx, dy) = (p[0] - mx, p[1] - my);
        a += dx * dx;
        b += dx * dy;
        c += dy * dy;
    }
    let lmin = (a + c) / 2.0 - (((a - c) / 2.0).powi(2) + b * b).sqrt();
    lmin.max(0.0).sqrt() / n.sqrt()
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Fit over every match, refining circle box sides, dropping the worst match while any is off by
/// more than OUTLIER_FT. Returns (H, inlier matches, errors).
pub fn robust_fit(matches: &[Match], fs: FrameSize) -> Option<(Mat3, Vec<Match>, Vec<f64>)> {
    let mut ms = matches.to_vec();
    let (h, err) = loop {
        if ms.len() < MIN_POINTS {
            return None;
        }
        if spread_off_line(&ms) < MIN_SPREAD_FT {
            return None;
        }
        let mut cur = fit(&ms);
        for _ in 0..REFINE_ITERATIONS {
            let Some((h, _)) = &cur else { break };
            let h = *h;
            refine_circle_edges(&mut ms, &h);
            cur = fit(&ms);
        }
        let (h, err) = cur?;
        let worst = argmax(&err);
        if err[worst] <= OUTLIER_FT || ms.len() == MIN_POINTS {
            break (h, err);
        }
        ms.remove(worst);
    };
    if err.iter().cloned().fold(f64::MIN, f64::max) > OUTLIER_FT || !plausible(&h, fs) {
        return None;
    }
    Some((h, ms, err))
}

/// Pick each line box's diagonal from where `h` predicts the line's ends; the agreeing corners.
fn resolve_lines(lines: &[LineCandidate], h: &Mat3, fs: FrameSize) -> Vec<Match> {
    let Some(h_inv) = invert(h) else {
        return vec![];
    };
    let k = keypoints();
    let mut out = Vec::new();
    for line in lines {
        let pred = |side: End| transform(&h_inv, k.xy(line.end(side)));
        let [x1, y1, x2, y2] = line.bx;
        let tol = LINE_END_TOL_PX.max(0.2 * (x2 - x1).hypot(y2 - y1));
        let mut best: Option<(bool, Vec<End>, f64)> = None;
        for slash in [true, false] {
            let corners = line.corners(slash, line.view, fs);
            let dists: Vec<(End, f64)> = corners
                .iter()
                .map(|(side, pt)| {
                    let p = pred(*side);
                    (*side, (pt[0] - p[0]).hypot(pt[1] - p[1]))
                })
                .collect();
            let good: Vec<End> = dists
                .iter()
                .filter(|(_, d)| *d < tol)
                .map(|(s, _)| *s)
                .collect();
            let sum: f64 = dists.iter().filter(|(_, d)| *d < tol).map(|(_, d)| d).sum();
            let better = !good.is_empty()
                && match &best {
                    None => true,
                    Some((_, bg, bs)) => {
                        good.len() > bg.len() || (good.len() == bg.len() && sum < *bs)
                    }
                };
            if better {
                best = Some((slash, good, sum));
            }
        }
        if let Some((slash, good, _)) = best {
            out.extend(line.matches(slash, line.view, fs, Some(&good)));
        }
    }
    out
}

/// Fit image -> rink (feet). Returns (H, inlier matches); see `compute_homography` in Python.
pub fn compute_homography(
    matches: &[Match],
    lines: &[LineCandidate],
    fs: FrameSize,
) -> (Option<Mat3>, Vec<Match>) {
    let first = robust_fit(matches, fs);
    if lines.is_empty() {
        return first.map_or((None, vec![]), |(h, inl, _)| (Some(h), inl));
    }
    if let Some((h, inliers, _)) = first {
        let extra = resolve_lines(lines, &h, fs);
        if !extra.is_empty() {
            let mut all = inliers.clone();
            all.extend(extra);
            if let Some((h2, inl2, _)) = robust_fit(&all, fs) {
                return (Some(h2), inl2);
            }
        }
        return (Some(h), inliers);
    }

    // No fit from the points alone: try each diagonal per line (itertools.product order).
    let n = lines.len();
    let mut best: (Option<Mat3>, Vec<Match>, f64) = (None, vec![], f64::INFINITY);
    for combo in 0..(1usize << n) {
        let mut all = matches.to_vec();
        for (i, line) in lines.iter().enumerate() {
            let slash = (combo >> (n - 1 - i)) & 1 == 0;
            all.extend(line.matches(slash, line.view, fs, None));
        }
        if all.len() <= MIN_POINTS {
            continue;
        }
        if let Some((h, inl, err)) = robust_fit(&all, fs) {
            let m = f32r(err.iter().sum::<f64>() / err.len() as f64);
            if inl.len() > MIN_POINTS && m < best.2 {
                best = (Some(h), inl, m);
            }
        }
    }
    (best.0, best.1)
}

/// Reject degenerate fits (mirrored, collapsed or absurdly scaled).
pub fn plausible(h: &Mat3, fs: FrameSize) -> bool {
    if h.determinant().abs() < 1e-12 {
        return false;
    }
    let (w, hh) = fs;
    let p0 = transform(h, [w / 2.0, hh / 2.0]);
    let p1 = transform(h, [w / 2.0 + 100.0, hh / 2.0]);
    let p2 = transform(h, [w / 2.0, hh / 2.0 - 100.0]);
    let right = [f32r(p1[0] - p0[0]), f32r(p1[1] - p0[1])];
    let up = [f32r(p2[0] - p0[0]), f32r(p2[1] - p0[1])];
    if right[0] * up[1] - right[1] * up[0] <= 0.0 {
        return false;
    }
    [right, up].iter().all(|v| {
        let n = v[0].hypot(v[1]);
        0.5 < n && n < 40.0
    })
}

/// Smooths H across frames and reuses it for a while when a frame has too few landmarks.
#[derive(Clone, Debug)]
pub struct HomographyTracker {
    pub smoothing: f64,
    pub max_age: usize,
    pub max_jump_ft: f64,
    pub h: Option<Mat3>,
    pub age: usize,
}

impl Default for HomographyTracker {
    fn default() -> Self {
        HomographyTracker {
            smoothing: 0.5,
            max_age: 10,
            max_jump_ft: 8.0,
            h: None,
            age: 0,
        }
    }
}

impl HomographyTracker {
    pub fn reset(&mut self) {
        self.h = None;
        self.age = 0;
    }

    fn jump_ft(old: &Mat3, new: &Mat3, fs: FrameSize) -> f64 {
        let (w, h) = fs;
        let mut worst = 0.0f64;
        for x in [0.2 * w, 0.5 * w, 0.8 * w] {
            for y in [0.4 * h, 0.7 * h] {
                let (a, b) = (transform(old, [x, y]), transform(new, [x, y]));
                worst = worst.max(f32r(f32r(a[0] - b[0]).hypot(f32r(a[1] - b[1]))));
            }
        }
        worst
    }

    pub fn update(&mut self, h_new: Option<Mat3>, fs: FrameSize) -> Option<Mat3> {
        if let Some(h_new) = h_new {
            let h_new = h_new / h_new[(2, 2)];
            self.h = Some(match self.h {
                Some(old) if Self::jump_ft(&old, &h_new, fs) <= self.max_jump_ft => {
                    old * self.smoothing + h_new * (1.0 - self.smoothing)
                }
                _ => h_new, // a camera cut or a big pan: start from the new fit
            });
            self.age = 0;
        } else if self.h.is_some() {
            self.age += 1;
            if self.age > self.max_age {
                self.reset();
            }
        }
        self.h
    }
}

/// Python's `round(x, ndigits)`: correctly rounded to that many decimals.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PlayerPosition {
    #[serde(rename = "class")]
    pub cls: String,
    pub confidence: f64,
    pub jersey_number: Option<String>,
    pub player_name: Option<String>,
    pub track_id: Option<i64>,
    pub box_xyxy: [f64; 4],
    pub rink_xy: [f64; 2],
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PuckPosition {
    pub confidence: f64,
    pub rink_xy: [f64; 2],
}

fn player_boxes(frame: &FrameRecord) -> Vec<Bx<'_>> {
    boxes(frame, "player", None)
        .into_iter()
        .filter(|b| PLAYER_CLASSES.contains(&b.det.cls.as_str()))
        .collect()
}

/// Each player's skate position (bottom-centre of the box) in rink feet.
pub fn project_players(frame: &FrameRecord, h: Option<&Mat3>) -> Vec<PlayerPosition> {
    let Some(h) = h else { return vec![] };
    player_boxes(frame)
        .into_iter()
        .filter_map(|p| {
            let [x, y] = transform(h, p.bottom_center());
            // Anything well outside the boards is a bad projection (or a player on the bench).
            (x.abs() <= rink::HALF_LENGTH + 5.0 && y.abs() <= rink::HALF_WIDTH + 5.0).then(|| {
                PlayerPosition {
                    cls: p.det.cls.clone(),
                    confidence: p.conf,
                    jersey_number: p.det.jersey_number.clone(),
                    player_name: p.det.player_name.clone(),
                    track_id: p.det.track_id,
                    box_xyxy: [p.x1, p.y1, p.x2, p.y2],
                    rink_xy: [
                        py_round(x.clamp(-rink::HALF_LENGTH, rink::HALF_LENGTH), 2),
                        py_round(y.clamp(-rink::HALF_WIDTH, rink::HALF_WIDTH), 2),
                    ],
                }
            })
        })
        .collect()
}

/// True when enough players are visible but they all land in an implausibly small patch.
pub fn players_clustered(players: &[PlayerPosition]) -> bool {
    if players.len() < MIN_PLAYERS_FOR_SPREAD {
        return false;
    }
    let n = players.len() as f64;
    let mx = players.iter().map(|p| p.rink_xy[0]).sum::<f64>() / n;
    let my = players.iter().map(|p| p.rink_xy[1]).sum::<f64>() / n;
    let ms = players
        .iter()
        .map(|p| (p.rink_xy[0] - mx).powi(2) + (p.rink_xy[1] - my).powi(2))
        .sum::<f64>()
        / n;
    ms.sqrt() < MIN_PLAYER_SPREAD_FT
}

/// Median player height in feet implied by `h`, or None with too few players.
pub fn implied_player_height(frame: &FrameRecord, h: Option<&Mat3>) -> Option<f64> {
    let h = h?;
    let bs = player_boxes(frame);
    if bs.len() < MIN_PLAYERS_FOR_HEIGHT {
        return None;
    }
    let mut heights: Vec<f64> = bs
        .iter()
        .map(|b| {
            let [x, y] = b.bottom_center();
            let (l, r) = (transform(h, [x - 20.0, y]), transform(h, [x + 20.0, y]));
            let d = f32r(f32r(r[0] - l[0]).hypot(f32r(r[1] - l[1])));
            (b.y2 - b.y1) * d / 40.0
        })
        .collect();
    heights.sort_by(f64::total_cmp);
    let n = heights.len();
    Some(if n % 2 == 1 {
        heights[n / 2]
    } else {
        (heights[n / 2 - 1] + heights[n / 2]) / 2.0
    })
}

/// The most confident puck detection in rink feet, or None.
pub fn project_puck(frame: &FrameRecord, h: Option<&Mat3>) -> Option<PuckPosition> {
    let h = h?;
    let pucks = boxes(frame, "puck", Some("puck"));
    let mut best = *pucks.first()?;
    for p in &pucks {
        if p.conf > best.conf {
            best = *p;
        }
    }
    let [x, y] = transform(h, best.center());
    if x.abs() > rink::HALF_LENGTH || y.abs() > rink::HALF_WIDTH {
        return None;
    }
    Some(PuckPosition {
        confidence: best.conf,
        rink_xy: [py_round(x, 2), py_round(y, 2)],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detections::parse;

    #[test]
    fn py_round_matches_python() {
        assert_eq!(py_round(1.005, 2), 1.0); // 1.005 is 1.00499999... in binary, as in Python
        assert_eq!(py_round(-12.3456, 2), -12.35);
        assert_eq!(py_round(2.675, 2), 2.67);
    }

    #[test]
    fn tracker_blends_and_expires() {
        let fs = (1280.0, 720.0);
        let h = Mat3::new(0.1, 0.0, -64.0, 0.0, -0.1, 36.0, 0.0, 0.0, 1.0);
        let mut t = HomographyTracker::default();
        assert_eq!(t.update(Some(h), fs), Some(h));
        for _ in 0..10 {
            assert!(t.update(None, fs).is_some());
        }
        assert!(t.update(None, fs).is_none());
    }

    #[test]
    fn frame_without_rink_markings_has_no_fit() {
        let frames = parse(r#"[{"frame": 0, "time_s": 0.0, "cut": false, "detections": {"player": [], "rink": [], "dots": []}}]"#).unwrap();
        let (m, l, v) = match_keypoints(&frames[0], (1280.0, 720.0));
        assert!(m.is_empty() && l.is_empty() && v == View::Side);
        assert!(compute_homography(&m, &l, (1280.0, 720.0)).0.is_none());
    }
}
