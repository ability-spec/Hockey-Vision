"""Write a synthetic detections.json (and a tiny placeholder video) for benchmarking and parity tests.

A pinhole broadcast camera high above the near boards pans along the rink and zooms in and out.
Rink markings (dots, circles, blue / centre / goal lines, the goal), players, goalies, referees
and the puck are projected through it into boxes with a little pixel noise, in the same format
run_models.py writes. No models, weights or real video are needed, so homography.py and the Rust
port can be timed and compared on exactly the same input.

    python rust/bench/synth_detections.py out_dir --frames 9000          # 5 minutes at 30 fps
    python homography.py out_dir/detections.json out_dir/placeholder.mp4

Only numpy is needed to write detections.json; the placeholder video (which homography.py opens
to read the frame size) needs OpenCV.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

W, H = 1280, 720
FPS = 30.0
HALF_LENGTH, HALF_WIDTH = 100.0, 42.5
CIRCLE_R = 15.0
DOT_R = 1.0

CAMERA = np.array([0.0, -120.0, 50.0])  # ft: centre ice, behind the near boards, up in the stands


def camera_basis(target):
    fwd = target - CAMERA
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return fwd, right, down


def project(points, cam):
    """(N, 3) rink points in ft -> (N, 2) pixels, and which are in front of the camera."""
    fwd, right, down, focal = cam
    rel = np.asarray(points, float) - CAMERA
    z = rel @ fwd
    u = W / 2 + focal * (rel @ right) / z
    v = H / 2 + focal * (rel @ down) / z
    return np.column_stack([u, v]), z > 1.0


def box_of(points, cam, rng, noise=1.5, min_visible=0.25):
    """Bounding box (clipped to the frame) of projected ground points, or None if mostly off-frame."""
    px, front = project(points, cam)
    if not front.all():
        return None
    x1, y1 = px.min(axis=0)
    x2, y2 = px.max(axis=0)
    area = max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < min_visible * area:
        return None
    jitter = rng.normal(0, noise, 4)
    b = [cx1 + jitter[0], cy1 + jitter[1], cx2 + jitter[2], cy2 + jitter[3]]
    b = [min(max(b[0], 0), W), min(max(b[1], 0), H), min(max(b[2], 0), W), min(max(b[3], 0), H)]
    return [round(v, 1) for v in b]


def ring(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a), np.zeros(n)])


def segment(x1, y1, x2, y2, n=16):
    t = np.linspace(0, 1, n)
    return np.column_stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, np.zeros(n)])


GOAL_LINE_BOARDS_Y = (HALF_WIDTH - 28) + math.sqrt(28 ** 2 - (89 - (HALF_LENGTH - 28)) ** 2)


def rink_detections(cam, rng):
    rink, dots = [], []

    def add(lst, cls, pts, conf=None, **kw):
        b = box_of(pts, cam, rng, **kw)
        if b is not None:
            lst.append({"class": cls, "confidence": round(conf or rng.uniform(0.5, 0.95), 4), "box_xyxy": b})

    add(dots, "dot", ring(0, 0, DOT_R, 12))
    for sx in (-1, 1):
        for sy in (-1, 1):
            add(dots, "dot", ring(sx * 20, sy * 22, DOT_R, 12))
            add(dots, "dot", ring(sx * 69, sy * 22, DOT_R, 12))
            add(rink, "Circle", ring(sx * 69, sy * 22, CIRCLE_R), min_visible=0.6)
        add(rink, "Blue_Line", segment(sx * 25, -HALF_WIDTH, sx * 25, HALF_WIDTH), min_visible=0.05)
        # The goal line is detected in two pieces, split at the goal.
        add(rink, "Goal_Back_Line", segment(sx * 89, 3.5, sx * 89, GOAL_LINE_BOARDS_Y), min_visible=0.05)
        add(rink, "Goal_Back_Line", segment(sx * 89, -GOAL_LINE_BOARDS_Y, sx * 89, -3.5), min_visible=0.05)
        goal = np.array([[sx * 89, y, z] for y in (-3, 3) for z in (0, 4)] +
                        [[sx * 92, y, 0] for y in (-3, 3)])
        add(rink, "Goal_Posts", goal, min_visible=0.8)
    add(rink, "Circle", ring(0, 0, CIRCLE_R), min_visible=0.6)
    add(rink, "Center_Line", segment(0, -HALF_WIDTH, 0, HALF_WIDTH), min_visible=0.05)
    rng.shuffle(rink)
    rng.shuffle(dots)
    return rink, dots


PEOPLE = ([("team_a_player", str(n)) for n in (8, 10, 24, 48, 63)] + [("goalie_a", "31")] +
          [("team_b_player", str(n)) for n in (11, 14, 22, 40, 77)] + [("goalie_b", "35")] +
          [("referee", None), ("referee", None)])


class Scene:
    def __init__(self, rng):
        self.rng = rng
        self.pos = np.column_stack([rng.uniform(-80, 80, len(PEOPLE)), rng.uniform(-35, 35, len(PEOPLE))])
        for i, (cls, _) in enumerate(PEOPLE):
            if cls.startswith("goalie"):
                self.pos[i] = [(-86 if cls == "goalie_a" else 86), 0]
        self.vel = rng.normal(0, 8, self.pos.shape)
        self.puck_owner = 0

    def step(self, dt):
        rng = self.rng
        self.vel += rng.normal(0, 6, self.vel.shape) * dt
        self.vel = np.clip(self.vel, -25, 25)
        for i, (cls, _) in enumerate(PEOPLE):
            if cls.startswith("goalie"):
                self.vel[i] *= 0.2
        self.pos += self.vel * dt
        self.pos[:, 0] = np.clip(self.pos[:, 0], -95, 95)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -40, 40)
        if rng.random() < 0.01:
            self.puck_owner = int(rng.integers(0, 12))

    def detections(self, cam, track_base):
        rng = self.rng
        players = []
        for i, (cls, number) in enumerate(PEOPLE):
            x, y = self.pos[i]
            (foot, head), front = project([[x, y, 0], [x, y, 6.0]], cam)
            if not front.all() or not (0 <= foot[0] < W and 0 <= foot[1] < H):
                continue
            hgt = foot[1] - head[1]
            box = [foot[0] - 0.22 * hgt, head[1], foot[0] + 0.22 * hgt, foot[1]] + rng.normal(0, 1.0, 4)
            det = {"class": cls, "confidence": round(rng.uniform(0.6, 0.95), 4),
                   "box_xyxy": [round(float(v), 1) for v in box], "track_id": track_base + i,
                   "jersey_number": number}
            players.append(det)
        puck = []
        if rng.random() < 0.6:
            x, y = self.pos[self.puck_owner] + [1.5, 0.5]
            (p,), front = project([[x, y, 0]], cam)
            if front.all() and 0 <= p[0] < W and 0 <= p[1] < H:
                puck.append({"class": "puck", "confidence": round(rng.uniform(0.6, 0.9), 4),
                             "box_xyxy": [round(float(p[0] - 3), 1), round(float(p[1] - 3), 1),
                                          round(float(p[0] + 3), 1), round(float(p[1] + 3), 1)]})
        return players, puck


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", help="Don't write placeholder.mp4")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    scene = Scene(rng)
    target_x, zoom, track_base = 0.0, 1.0, 1
    pan_speed = 15.0
    with open(args.out / "detections.json", "w") as f:
        f.write("[")
        for idx in range(args.frames):
            cut = idx > 0 and rng.random() < 1 / 400
            if cut:  # a new shot: somewhere else on the ice, new track ids
                target_x = float(rng.uniform(-70, 70))
                zoom = float(rng.choice([1.0, 1.0, 1.0, 1.6, 2.6]))
                track_base += 100
            # The camera follows the play, roughly.
            follow = float(np.clip(scene.pos[scene.puck_owner, 0], -70, 70))
            target_x += float(np.clip(follow - target_x, -pan_speed / FPS, pan_speed / FPS))
            cam_target = np.array([target_x, 4.0, 0.0])
            fwd, right, down = camera_basis(cam_target)
            cam = (fwd, right, down, 1250.0 * zoom)

            rink, dots = rink_detections(cam, rng)
            players, puck = scene.detections(cam, track_base)
            scene.step(1 / FPS)
            record = {"frame": idx, "time_s": round(idx / FPS, 3), "cut": bool(cut),
                      "detections": {"player": players, "puck": puck, "rink": rink, "dots": dots, "number": []}}
            f.write(("\n" if idx == 0 else ",\n") + json.dumps(record))
        f.write("\n]\n")

    if not args.no_video:
        import cv2
        out = cv2.VideoWriter(str(args.out / "placeholder.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        out.write(np.zeros((H, W, 3), np.uint8))
        out.release()
    print(f"Wrote {args.frames} frames to {args.out / 'detections.json'}")


if __name__ == "__main__":
    main()
