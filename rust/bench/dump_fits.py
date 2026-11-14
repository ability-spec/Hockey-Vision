"""Each frame's own fit before smoothing, one JSON line per frame; the Python side of
rust/crates/hv-core/examples/dump_fits.rs.

    python rust/bench/dump_fits.py <detections.json> <W>x<H> > fits.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_python_homography  # noqa: E402,F401  (adds the singular-fit guard)
import homography as hg  # noqa: E402

frames = json.loads(Path(sys.argv[1]).read_text())
w, h = map(int, sys.argv[2].split("x"))
for f in frames:
    m, lines, _ = hg.match_keypoints(f["detections"], (w, h))
    H, inliers = hg.compute_homography(m, lines, (w, h))
    print(json.dumps({"frame": f["frame"], "H": None if H is None else (H / H[2, 2]).tolist(),
                      "inliers": [x.keypoint for x in inliers]}))
