"""How much of run_models.py's per-frame time is Python around the models rather than the models.

No trained weights are needed: a randomly initialised YOLO26n (built from Ultralytics' yolo26n.yaml)
has the same architecture and so the same cost as the five Hockey-Vision models. For each kind of
call run_models.py makes it times:

    full-frame model   model.predict() on the frame downscaled to 640x360 (as run_on_frame does),
                       split into Ultralytics' own preprocess / inference / postprocess, against a
                       bare forward pass of the network on an already prepared tensor
    player tracking    model.track() (ByteTrack) instead of predict()
    number model       predict() on a batch of player crops (upscaled to 640, as _read_jerseys does)
    per-frame Python   results_to_dicts() + json.dumps() for a frame of 14 players, a puck and the
                       rink / dot boxes, and the jersey vote bookkeeping

    python rust/bench/ultralytics_overhead.py [--device cpu] [--frames 60]

Everything runs on whichever device is given; on a CPU the forward passes are far slower than on
a GPU, but the Python overhead (what a Rust runner removes) is what this measures.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run_models  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.results import Results  # noqa: E402


def timed(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return statistics.median(out)


def speed_of(fn, n, warmup=3):
    """Median of Ultralytics' own per-image speed dict over n calls."""
    for _ in range(warmup):
        fn()
    parts = {"preprocess": [], "inference": [], "postprocess": []}
    for _ in range(n):
        r = fn()[0]
        for k in parts:
            parts[k].append(r.speed[k])
    return {k: statistics.median(v) for k, v in parts.items()}


def fake_player_result(frame, rng, n=14):
    names = {0: "goalie_a", 1: "goalie_b", 2: "puck", 3: "referee", 4: "team_a_player", 5: "team_b_player"}
    xy = rng.uniform([0, 200], [1800, 900], (n, 2))
    boxes = np.column_stack([xy, xy + [60, 150], rng.uniform(0.5, 0.9, n), np.arange(1, n + 1),
                             rng.choice([4, 5], n)]).astype(np.float32)
    return Results(frame, path="", names=names, boxes=torch.from_numpy(boxes))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--threads", type=int, default=0, help="torch threads (default: torch's choice)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    small = np.ascontiguousarray(frame[::3, ::3][:360, :640])  # 640x360, what the full-frame models get

    model = YOLO("yolo26n.yaml", task="detect")
    model.predict(small, imgsz=[384, 640], device=args.device, verbose=False)  # builds the predictor
    net = model.predictor.model
    out = {"device": args.device, "torch_threads": torch.get_num_threads()}

    # Full-frame model: predict() vs the bare network on a prepared tensor.
    predict = lambda: model.predict(small, imgsz=[384, 640], conf=0.25, device=args.device, verbose=False)
    out["full_frame_predict_ms"] = timed(lambda: predict()[0].cpu(), args.frames)
    out["full_frame_speed_ms"] = speed_of(predict, args.frames)
    x = torch.zeros(1, 3, 384, 640, device=args.device)
    with torch.inference_mode():
        out["full_frame_forward_ms"] = timed(lambda: net(x), args.frames)

    # Tracking (ByteTrack in Python) on top of predict.
    track = lambda: model.track(small, imgsz=[384, 640], conf=0.25, device=args.device, verbose=False,
                                persist=True, tracker="bytetrack.yaml")[0].cpu()
    out["player_track_ms"] = timed(track, args.frames)

    # Number model: a batch of player crops, upscaled to 640.
    crops = [np.ascontiguousarray(frame[200:350, 100 * i:100 * i + 60]) for i in range(14)]
    batch = lambda: [r.cpu() for r in model.predict(crops, imgsz=640, conf=0.25, device=args.device, verbose=False)]
    out["number_batch14_predict_ms"] = timed(batch, max(5, args.frames // 4))
    out["number_batch14_speed_ms_per_crop"] = speed_of(
        lambda: model.predict(crops, imgsz=640, conf=0.25, device=args.device, verbose=False), max(5, args.frames // 4))
    xb = torch.zeros(14, 3, 640, 640, device=args.device)
    with torch.inference_mode():
        out["number_batch14_forward_ms"] = timed(lambda: net(xb), max(5, args.frames // 4))

    # Per-frame Python bookkeeping in run_models.py: detections to dicts and JSON, jersey votes.
    players = fake_player_result(frame, rng)
    others = {name: fake_player_result(frame, rng, n) for name, n in (("puck", 1), ("rink", 6), ("dots", 3))}
    results = {"player": players, **others}
    jerseys = [{"digits": [], "reading": ("12", 0.8), "number": "12"} for _ in range(14)]

    def bookkeeping():
        record = {"frame": 1, "time_s": 0.033, "cut": False,
                  "detections": run_models.results_to_dicts(results, jerseys, None, 3.0)}
        json.dumps(record)
    out["results_to_dicts_and_json_ms"] = timed(bookkeeping, 200)

    votes = run_models.JerseyVotes()
    keys = [(i, "team_a_player") for i in range(14)]

    def voting():
        for k in keys:
            votes.add(k, ("12", 0.8))
            votes.settled(k)
            votes.number(k)
    out["jersey_votes_ms"] = timed(voting, 200)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
