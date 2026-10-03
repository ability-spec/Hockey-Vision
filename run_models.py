"""Run the Hockey-Vision YOLO models on an image, a video, or a folder of either.

Examples:
    python run_models.py path/to/frame.jpg
    python run_models.py path/to/game.mp4 --models player puck --conf 0.3
    python run_models.py path/to/folder --no-per-model --output results

For every input this writes to <output>/<input name>/:
    combined.<ext>        all selected models drawn on one image/video
    <model>.<ext>         one annotated output per model (unless --no-per-model)
    detections.json       every detection (per frame for videos)
With --no-video only detections.json is written.

Each model is loaded from CV_Models/<model>_model.engine when that exists (a TensorRT engine
built by export_engines.py, for the Jetson), otherwise from CV_Models/<model>_model.pt.

Jersey numbers: the number model detects single digits, so it is run on each player's box
(upscaled) and the digits found are joined into the player's number. In videos players are
tracked across frames, and each track's number is a vote over every frame's reading, so it
stays with the player through frames where it can't be read and one-off misreads are outvoted.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "CV_Models"
MODEL_NAMES = ["player", "puck", "number", "rink", "dots"]

# BGR colour per model for the combined output, so each model's boxes are distinguishable.
MODEL_COLORS = {
    "player": (60, 180, 75),
    "puck": (0, 0, 255),
    "number": (0, 215, 255),
    "rink": (255, 130, 0),
    "dots": (240, 50, 230),
}

# Classes to drop from a model's output. The puck model handles pucks, so the player model's
# puck class is ignored to avoid duplicate boxes.
EXCLUDED_CLASSES = {
    "player": {"puck"},
}

# Per-model confidence thresholds that differ from --conf. The puck is small and often blurred,
# so its detections score lower and a lower threshold keeps more of them.
MODEL_CONF = {
    "puck": 0.10,
}

# Jersey numbers.
NUMBER_IMGSZ = 640          # player crops are upscaled to this for the digit model
# Player crops per digit-model call; export_engines.py builds the engine for up to this many.
# 16 needed more memory than an 8 GB Orin Nano has free to build the engine; 4 builds fine.
NUMBER_BATCH = 4
MIN_NUMBER_CROP_PX = 60     # players shorter than this are too small to read a number from
DIGIT_NMS_IOU = 0.5         # overlapping digit boxes of different classes: keep the most confident
# Jersey numbers sit on the torso: digit centres are 20-60% of the way down the player's box and
# digits are 6-30% of its height. Digits elsewhere are stripes on socks, sticks or the boards.
DIGIT_Y_RANGE = (0.1, 0.62)
DIGIT_HEIGHT_RANGE = (0.06, 0.3)
MIN_READING_CONF = 0.4      # weaker readings don't vote
MIN_JERSEY_VOTES = 1.2      # summed confidence a track's number needs before it is shown
PARTIAL_READ_WEIGHT = 0.5   # a one-digit read of a two-digit number (e.g. "2" of "12") counts this much
SWITCH_RATIO = 1.5          # a track's shown number only changes when another scores this much higher

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_path(name, device=None):
    """TensorRT only for CUDA; CPU/MPS always use portable .pt weights."""
    engine = MODELS_DIR / f"{name}_model.engine"
    selected = str(pick_device(device)).lower()
    cuda = selected.startswith("cuda") or selected.isdecimal()
    return engine if cuda and engine.exists() else MODELS_DIR / f"{name}_model.pt"


def load_model(name, device=None):
    path = model_path(name, device)
    if not path.exists():
        sys.exit(f"Model not found: {path}")
    return YOLO(str(path), task="detect")


def load_models(names, device=None):
    for name in names:
        print(f"  {name}: {model_path(name, device).name}")
    return {name: load_model(name, device) for name in names}


def kept_classes(name, model):
    """Class ids to keep for a model, or None to keep all of them."""
    excluded = EXCLUDED_CLASSES.get(name)
    if not excluded:
        return None
    return [i for i, cls in model.names.items() if cls not in excluded]


def run_on_frame(models, frame, conf, imgsz, device, track=False):
    """Return {model_name: ultralytics Results} for a single BGR frame.

    `conf` is {model_name: threshold}. The number model isn't run here: it reads digits off the
    player boxes, see read_jerseys(). With `track`, players get track ids that persist across
    calls (one video at a time).
    """
    out = {}
    for name, model in models.items():
        if name == "number":
            continue
        run = model.track if track and name == "player" else model.predict
        kwargs = {"persist": True, "tracker": "bytetrack.yaml"} if run == model.track else {}
        out[name] = run(frame, conf=conf[name], imgsz=imgsz, device=device,
                        classes=kept_classes(name, model), verbose=False, **kwargs)[0]
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def assemble_number(digits):
    """Join the digits found on one player into a jersey number. Returns (number, confidence)
    or None.

    Starts from the most confident digit and adds at most one neighbour: a digit of about the
    same height, level with it and next to it. Anything else in the box (a sleeve number, a
    stray read on a logo) is ignored.
    """
    kept = []
    for d in sorted(digits, key=lambda d: -d["confidence"]):
        if all(_iou(d["box_xyxy"], k["box_xyxy"]) < DIGIT_NMS_IOU for k in kept):
            kept.append(d)
    if not kept:
        return None
    best = kept[0]
    x1, y1, x2, y2 = best["box_xyxy"]
    h, cy = y2 - y1, (y1 + y2) / 2
    group = [best]
    for d in kept[1:]:
        dx1, dy1, dx2, dy2 = d["box_xyxy"]
        gap = max(dx1 - x2, x1 - dx2)  # horizontal space between the boxes, negative if they overlap
        if (abs((dy2 - dy1) - h) < 0.35 * h and abs((dy1 + dy2) / 2 - cy) < 0.5 * h
                and -0.3 * h < gap < 0.6 * h):
            group.append(d)
            break
    group.sort(key=lambda d: d["box_xyxy"][0])
    number = "".join(d["class"] for d in group)
    if number.startswith("0"):  # the NHL doesn't allow 0 or 00; a lone "0" is usually a logo
        return None
    return number, sum(d["confidence"] for d in group) / len(group)


def read_jerseys(number_model, frame, player_result, conf, device):
    """Read each player's number this frame. Returns one dict per player box, in box order:
    {"digits": digit detections in frame coordinates, "reading": (number, conf) or None,
    "number": the number to show, filled in by the caller}."""
    jerseys = [{"digits": [], "reading": None, "number": None} for _ in range(len(player_result.boxes))]
    crops, offsets = [], []
    h, w = frame.shape[:2]
    for i, box in enumerate(player_result.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
        if y2 - y1 < MIN_NUMBER_CROP_PX or x2 <= x1:
            continue
        crops.append(frame[y1:y2, x1:x2])
        offsets.append((i, x1, y1))
    if not crops:
        return jerseys
    players = player_result.boxes.xyxy.tolist()
    results = []
    for start in range(0, len(crops), NUMBER_BATCH):
        results += number_model.predict(crops[start:start + NUMBER_BATCH], conf=conf, imgsz=NUMBER_IMGSZ,
                                        device=device, verbose=False)
    for (i, ox, oy), r in zip(offsets, results):
        box_h = players[i][3] - players[i][1]
        for box, cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if not (DIGIT_Y_RANGE[0] <= (box[1] + box[3]) / 2 / box_h <= DIGIT_Y_RANGE[1]
                    and DIGIT_HEIGHT_RANGE[0] <= (box[3] - box[1]) / box_h <= DIGIT_HEIGHT_RANGE[1]):
                continue
            # Where players overlap, a digit inside both boxes could be either one's.
            cx, cy = (box[0] + box[2]) / 2 + ox, (box[1] + box[3]) / 2 + oy
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for k, b in enumerate(players) if k != i):
                continue
            jerseys[i]["digits"].append({
                "class": r.names[int(cls)],
                "confidence": round(score, 4),
                "box_xyxy": [round(v + o, 1) for v, o in zip(box, (ox, oy, ox, oy))],
            })
        jerseys[i]["reading"] = assemble_number(jerseys[i]["digits"])
    return jerseys


class JerseyVotes:
    """Per player track (any hashable key), the summed confidence of every number read on it. Once a track's number
    is shown it sticks to the player, whether or not the number is readable in later frames."""

    def __init__(self):
        self.votes = defaultdict(Counter)
        self.shown = {}

    def add(self, track, reading):
        if track is not None and reading is not None and reading[1] >= MIN_READING_CONF:
            self.votes[track][reading[0]] += reading[1]

    def number(self, track):
        votes = self.votes.get(track)
        if not votes:
            return None

        def score(n):
            # A two-digit number is also backed by reads that only caught one of its digits.
            partial = sum(votes[d] for d in set(n) if d in votes) if len(n) == 2 else 0.0
            return votes[n] + PARTIAL_READ_WEIGHT * partial

        best = max(votes, key=score)
        current = self.shown.get(track)
        if current is not None and score(best) < SWITCH_RATIO * score(current):
            return current
        if score(best) >= MIN_JERSEY_VOTES:
            self.shown[track] = best
        return self.shown.get(track)


def track_ids(result):
    ids = result.boxes.id
    return [None] * len(result.boxes) if ids is None else [int(i) for i in ids.tolist()]


def results_to_dicts(results, jerseys=None):
    out = {}
    for name, r in results.items():
        dets = []
        ids = track_ids(r)
        for i, (box, cls, score) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                                   r.boxes.conf.tolist())):
            det = {
                "class": r.names[int(cls)],
                "confidence": round(score, 4),
                "box_xyxy": [round(v, 1) for v in box],
            }
            if name == "player":
                if ids[i] is not None:
                    det["track_id"] = ids[i]
                if jerseys is not None:
                    reading = jerseys[i]["reading"]
                    det["jersey_number"] = jerseys[i]["number"]
                    det["jersey_reading"] = None if reading is None else {
                        "number": reading[0], "confidence": round(reading[1], 4)}
            dets.append(det)
        out[name] = dets
    if jerseys is not None:
        out["number"] = [d for j in jerseys for d in j["digits"]]
    return out


def draw_label(img, text, x, y, color):
    scale = max(0.4, img.shape[1] / 2500)
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y = max(y, th + baseline)
    cv2.rectangle(img, (x, y - th - baseline), (x + tw, y), color, -1)
    cv2.putText(img, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_jersey(img, number, box, color):
    """A "#12" badge centred above the player's box (above its class label), clear of the jersey."""
    x1, y1, x2, _ = box
    text = f"#{number}"
    scale = max(0.6, img.shape[1] / 1600)
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    label_scale = max(0.4, img.shape[1] / 2500)  # same as draw_label
    (_, lh), lb = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, label_scale, max(1, int(label_scale * 2)))
    label_h = lh + lb
    x = int((x1 + x2) / 2 - tw / 2)
    y = max(th + baseline + 4, int(y1) - label_h - 4)
    cv2.rectangle(img, (x - 4, y - th - baseline - 4), (x + tw + 4, y + 2), color, -1)
    cv2.rectangle(img, (x - 4, y - th - baseline - 4), (x + tw + 4, y + 2), (255, 255, 255), 1)
    cv2.putText(img, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_numbers(frame, results, jerseys, digits=True):
    """Digit boxes and each player's jersey number badge."""
    img = frame.copy()
    color = MODEL_COLORS["number"]
    line = max(1, int(img.shape[1] / 800))
    for box, j in zip(results["player"].boxes.xyxy.tolist(), jerseys):
        if digits:
            for d in j["digits"]:
                x1, y1, x2, y2 = map(int, d["box_xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, line)
        if j["number"] is not None:
            draw_jersey(img, j["number"], box, MODEL_COLORS["player"])
    return img


def draw_combined(frame, results, jerseys=None):
    img = frame.copy()
    line = max(1, int(img.shape[1] / 800))
    for name, r in results.items():
        color = MODEL_COLORS.get(name, (200, 200, 200))
        ids = track_ids(r)
        for box, cls, score, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                        r.boxes.conf.tolist(), ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line)
            label = f"{r.names[int(cls)]} {score:.2f}" + (f" id{tid}" if tid is not None else "")
            draw_label(img, label, x1, y1, color)
    if jerseys is not None:
        img = draw_numbers(img, results, jerseys)

    # Legend in the top-left corner.
    y = 10
    for name in list(results) + (["number"] if jerseys is not None else []):
        y += 25
        draw_label(img, name, 10, y, MODEL_COLORS.get(name, (200, 200, 200)))
    return img


def per_model_frame(name, frame, results, jerseys):
    if name == "number":
        return draw_numbers(frame, results, jerseys)
    return results[name].plot()


def process_image(path, models, args, out_dir):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"  Could not read {path}, skipping")
        return
    results = run_on_frame(models, frame, args.model_conf, args.imgsz, args.device)
    jerseys = None
    if "number" in models:
        # A single image has nothing to vote over, so each player shows this frame's reading.
        jerseys = read_jerseys(models["number"], frame, results["player"], args.model_conf["number"], args.device)
        for j in jerseys:
            j["number"] = j["reading"][0] if j["reading"] else None

    if args.write_video:
        cv2.imwrite(str(out_dir / f"combined{path.suffix}"), draw_combined(frame, results, jerseys))
    if args.write_video and args.per_model:
        for name in models:
            cv2.imwrite(str(out_dir / f"{name}{path.suffix}"), per_model_frame(name, frame, results, jerseys))

    detections = results_to_dicts(results, jerseys)
    (out_dir / "detections.json").write_text(json.dumps(detections, indent=2))
    counts = ", ".join(f"{n}: {len(d)}" for n, d in detections.items())
    print(f"  {counts}")


def process_video(path, models, args, out_dir):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {}
    if args.write_video:
        writers["combined"] = cv2.VideoWriter(str(out_dir / "combined.mp4"), fourcc, fps, (w, h))
        if args.per_model:
            for name in models:
                writers[name] = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), fourcc, fps, (w, h))

    if "player" in models:
        # A fresh player model per video, so track ids (and the tracker's state) start over.
        models = dict(models, player=load_model("player", args.device))
    votes = JerseyVotes()

    all_detections = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and idx >= args.max_frames):
                break
            results = run_on_frame(models, frame, args.model_conf, args.imgsz, args.device, track=True)
            jerseys = None
            if "number" in models:
                jerseys = read_jerseys(models["number"], frame, results["player"],
                                       args.model_conf["number"], args.device)
                # The tracker sometimes hands an id over to a nearby player; keying on the class
                # too means a handover to the other team (or a referee) doesn't inherit the number.
                r = results["player"]
                keys = [None if tid is None else (tid, r.names[int(cls)])
                        for tid, cls in zip(track_ids(r), r.boxes.cls.tolist())]
                for key, j in zip(keys, jerseys):
                    votes.add(key, j["reading"])
                for key, j in zip(keys, jerseys):
                    j["number"] = votes.number(key)

            if "combined" in writers:
                writers["combined"].write(draw_combined(frame, results, jerseys))
            for name in models:
                if name in writers:
                    writers[name].write(per_model_frame(name, frame, results, jerseys))

            all_detections.append({"frame": idx, "time_s": round(idx / fps, 3),
                                   "detections": results_to_dicts(results, jerseys)})
            idx += 1
            if idx % 25 == 0 or idx == total:
                print(f"\r  frame {idx}/{total or '?'}", end="", flush=True)
    finally:
        cap.release()
        for wr in writers.values():
            wr.release()
    print()

    if not all_detections:
        raise RuntimeError(f"No frames decoded from video: {path}")
    (out_dir / "detections.json").write_text(json.dumps(all_detections, indent=2))


def collect_inputs(source):
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
    return [source]


def main():
    parser = argparse.ArgumentParser(description="Run the Hockey-Vision models and save annotated outputs.")
    parser.add_argument("source", type=Path, help="Image, video, or folder of images/videos")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES,
                        help="Which models to run (default: all)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for every model except the puck (default: 0.25)")
    parser.add_argument("--puck-conf", type=float, default=MODEL_CONF["puck"],
                        help=f"Confidence threshold for the puck model (default: {MODEL_CONF['puck']})")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, 0... (default: auto)")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output folder (default: outputs)")
    parser.add_argument("--no-per-model", dest="per_model", action="store_false",
                        help="Only write the combined output, not one per model")
    parser.add_argument("--no-video", dest="write_video", action="store_false",
                        help="Only write detections.json, no annotated images/videos (faster)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop videos after N frames (0 = all)")
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"Source not found: {args.source}")
    args.device = pick_device(args.device)
    args.model_conf = {name: MODEL_CONF.get(name, args.conf) for name in MODEL_NAMES}
    args.model_conf["puck"] = args.puck_conf

    if "number" in args.models and "player" not in args.models:
        args.models = args.models + ["player"]  # jersey numbers are read off the player boxes
        print("Adding the player model: the number model needs player boxes")

    inputs = collect_inputs(args.source)
    if not inputs:
        sys.exit(f"No images or videos found in {args.source}")

    print(f"Loading models: {', '.join(args.models)} (device: {args.device})")
    models = load_models(args.models, args.device)

    for path in inputs:
        out_dir = args.output / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{path} -> {out_dir}/")
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            process_image(path, models, args, out_dir)
        elif ext in VIDEO_EXTS:
            process_video(path, models, args, out_dir)
        else:
            print(f"  Unsupported file type {ext}, skipping")


if __name__ == "__main__":
    main()

