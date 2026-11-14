"""Check hv_core::image against OpenCV on real video frames, value for value: decoding (yuv420p ->
BGR as cv2.VideoCapture gives it), cv2.resize INTER_LINEAR (the 1/3 frame downscale and random
player-crop resizes) and CutDetector's histograms.

    python rust/bench/check_pixels.py <video> [frames]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from video_io import CutDetector  # noqa: E402


def main():
    video, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cap = cv2.VideoCapture(video)
    frames = [cap.read()[1] for _ in range(n)]
    h, w = frames[0].shape[:2]
    tmp = Path(tempfile.mkdtemp())
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-frames:v", str(n), "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    (tmp / "f.yuv").write_bytes(raw)
    r = random.Random(0)
    crops = []
    for _ in range(300):
        f = r.randrange(n)
        x1, y1 = r.randint(0, w - 2), r.randint(0, h - 2)
        x2, y2 = min(w, x1 + r.randint(1, 250)), min(h, y1 + r.randint(1, 320))
        cw, ch = x2 - x1, y2 - y1
        g = min(640 / ch, 640 / cw)
        dw, dh = (round(cw * g), round(ch * g)) if r.random() < 0.8 else (max(1, cw // 3), max(1, ch // 2))
        crops.append((f, x1, y1, x2, y2, dw, dh))
    (tmp / "crops.txt").write_text("\n".join(" ".join(map(str, c)) for c in crops))
    exe = ROOT / "rust" / "target" / "release" / "examples" / "check_pixels"
    subprocess.run([str(exe), str(tmp / "f.yuv"), str(w), str(h), str(n), str(tmp / "crops.txt"), str(tmp)], check=True)

    bgr = np.frombuffer((tmp / "bgr.raw").read_bytes(), np.uint8).reshape(n, h, w, 3)
    print(f"decode:            {sum(int((bgr[k] != frames[k]).sum()) for k in range(n))} values differ of {bgr.size}")
    small = np.frombuffer((tmp / "small.raw").read_bytes(), np.uint8).reshape(n, 360, 640, 3)
    ref = [cv2.resize(f, (640, 360), interpolation=cv2.INTER_LINEAR) for f in frames]
    print(f"resize 1/3:        {sum(int((small[k] != ref[k]).sum()) for k in range(n))} values differ of {small.size}")
    data, off, bad, tot = (tmp / "crops.raw").read_bytes(), 0, 0, 0
    for f, x1, y1, x2, y2, dw, dh in crops:
        mine = np.frombuffer(data[off:off + dw * dh * 3], np.uint8).reshape(dh, dw, 3)
        off += dw * dh * 3
        theirs = cv2.resize(np.ascontiguousarray(frames[f][y1:y2, x1:x2]), (dw, dh), interpolation=cv2.INTER_LINEAR)
        bad += int((mine != theirs).sum())
        tot += mine.size
    print(f"resize crops:      {bad} values differ of {tot} ({len(crops)} crops)")
    hists = np.frombuffer((tmp / "hists.raw").read_bytes(), np.float32).reshape(n, 32, 32)
    det = CutDetector()
    worst = 0.0
    for k in range(n):
        det.update(frames[k])
        worst = max(worst, float(np.abs(det.prev - hists[k]).max()))
    print(f"cut histograms:    max difference {worst:.3g} (values are 0..1)")


if __name__ == "__main__":
    main()
