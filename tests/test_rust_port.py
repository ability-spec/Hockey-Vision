"""The Rust port (rust/) against the Python pipeline it replaces.

Every Rust component that can match Python exactly is checked exactly: export_data.py's files,
Ultralytics' NMS and ByteTrack, roster.py, the jersey-number assembly and votes, CPython's
round / hypot / divmod, and OpenCV's decoding, resizes and cut histograms. The two stages that
can't be exact are checked with tolerances: the rink fit (the Rust fit follows OpenCV 5's solver)
and the detection loop (ONNX Runtime versions differ in float rounding).

    python -m unittest tests.test_rust_port -v

Needs cargo (rust/ is built on first run) and ffmpeg. The detection tests also need the ONNX
models (python rust/export_onnx.py) and a video (HV_TEST_VIDEO, default
outputs/speedup_test/clips/window1_02m48s.mp4); they are skipped without them.
"""
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUST = ROOT / "rust"
RELEASE = RUST / "target" / "release"
BENCH = RUST / "bench"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))
PY = sys.executable
MODELS = ROOT / "CV_Models" / "Models" / "onnx"
VIDEO = Path(os.environ.get("HV_TEST_VIDEO", ROOT / "outputs/speedup_test/clips/window1_02m48s.mp4"))
ROSTER = ROOT / "rosters" / "2026-03-03_MTL_at_SJS.csv"
_built = None


def cargo():
    return shutil.which("cargo") or (str(Path.home() / ".cargo/bin/cargo") if (Path.home() / ".cargo/bin/cargo").exists() else None)


def setUpModule():
    """Build the binary and the test helpers once (a no-op when they're up to date)."""
    global _built
    exe = cargo()
    if exe is None:
        raise unittest.SkipTest("cargo not found: install Rust (rustup.rs) to test the Rust port")
    features = ["--features", "coreml"] if sys.platform == "darwin" else []
    for args in (["build", "--release", *features], ["build", "--release", "--examples", *features]):
        r = subprocess.run([exe, *args], cwd=RUST, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"cargo {' '.join(args)} failed:\n{r.stderr[-3000:]}")
    _built = RELEASE / "hockey-vision"


def hv(*args, **kw):
    return subprocess.run([str(_built), *map(str, args)], capture_output=True, text=True, check=True, **kw)


def parity(req):
    """Run one hv_core component through rust/crates/hv-core/examples/parity.rs."""
    r = subprocess.run([str(RELEASE / "examples" / "parity")], input=json.dumps(req), capture_output=True,
                       text=True, check=True)
    return json.loads(r.stdout)


def run_py(*args):
    return subprocess.run([PY, *map(str, args)], cwd=ROOT, capture_output=True, text=True, check=True)


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hv-rust-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# ---- exact: export_data.py / events.py

class ExportTests(TempDir):
    def export_both(self, clips, name):
        run_py("export_data.py", *clips, "--output", self.tmp / f"py_{name}")
        hv("export", *clips, "--output", self.tmp / f"rs_{name}")
        from compare_exports import differing_files
        return differing_files(self.tmp / f"py_{name}", self.tmp / f"rs_{name}")

    def test_synthetic_games_byte_identical(self):
        """Thousands of possessions, shots, passes, turnovers and defensive plays."""
        for seed in (0, 1):
            clip = self.tmp / f"game{seed}"
            run_py(BENCH / "synth_game.py", clip, "--frames", 6000, "--seed", seed)
            self.assertEqual(self.export_both([clip], f"g{seed}"), [], f"seed {seed}")
            events = (self.tmp / f"rs_g{seed}" / "events.csv").read_text().count("\n")
            self.assertGreater(events, 50, "the synthetic game should produce plenty of events")

    def test_several_clips_byte_identical(self):
        clips = []
        for seed in (2, 3):
            clips.append(self.tmp / f"clip{seed}")
            run_py(BENCH / "synth_game.py", clips[-1], "--frames", 1500, "--seed", seed)
        self.assertEqual(self.export_both(clips, "multi"), [])

    def test_without_positions_byte_identical(self):
        clip = self.tmp / "nopos"
        run_py(BENCH / "synth_game.py", clip, "--frames", 600, "--seed", 4)
        (clip / "positions.json").unlink()
        self.assertEqual(self.export_both([clip], "nopos"), [])


class PyMathTests(TempDir):
    def test_hypot_and_divmod_bit_exact(self):
        """pyval.rs reproduces this Python's math.hypot and float divmod to the last bit."""
        r = random.Random(1)
        with open(self.tmp / "v.bin", "wb") as f:
            for _ in range(100_000):
                x = r.uniform(-200, 200) if r.random() < 0.5 else r.gauss(0, 1) * 10 ** r.randint(-8, 8)
                y = r.uniform(-200, 200) if r.random() < 0.5 else r.gauss(0, 1) * 10 ** r.randint(-8, 8)
                t = r.uniform(0, 4000)
                d, m = divmod(t, 60.0)
                f.write(struct.pack("<6d", x, y, math.hypot(x, y), t, d, m))
        out = subprocess.run([str(RELEASE / "examples" / "check_pymath"), str(self.tmp / "v.bin")],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("hypot differs on 0 ", out)
        self.assertIn("divmod on 0", out)


# ---- exact: pixels (OpenCV)

class PixelTests(TempDir):
    def test_decode_resize_histograms_match_opencv(self):
        """A generated yuv420p video: decoding, the 1/3 downscale, 300 crop resizes, histograms."""
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not found")
        video = self.tmp / "test.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
                        "-frames:v", "4", "-pix_fmt", "yuv420p", str(video)], check=True)
        from check_pixels import check
        r = check(str(video), 4)
        self.assertEqual((r["decode"], r["resize"], r["crops"]), (0, 0, 0))
        self.assertLess(r["hist"], 1e-6)


# ---- exact: Ultralytics NMS and ByteTrack

class NmsTests(unittest.TestCase):
    def test_matches_ultralytics(self):
        import torch
        import torchvision  # noqa: F401  (Ultralytics uses torchvision's NMS when it's imported)
        from ultralytics.utils import nms as unms, ops
        rng = np.random.default_rng(0)
        for trial in range(30):
            nc, anchors = int(rng.integers(1, 8)), int(rng.integers(50, 600))
            # Boxes clustered around a few centres so plenty overlap.
            centres = rng.uniform([0, 0], [640, 384], (8, 2))
            xy = centres[rng.integers(0, 8, anchors)] + rng.normal(0, 15, (anchors, 2))
            wh = rng.uniform(5, 120, (anchors, 2))
            scores = rng.random((anchors, nc)) ** 4
            pred = np.concatenate([xy, wh, scores], 1).T.astype(np.float32)  # (4 + nc, anchors)
            conf = float(rng.choice([0.25, 0.6]))
            classes = sorted(rng.choice(nc, max(1, nc - 1), replace=False).tolist()) if nc > 1 and trial % 2 else None
            image = (360, 640) if trial % 3 else (int(rng.integers(60, 300)), int(rng.integers(20, 200)))
            inp = (384, 640) if trial % 3 else (640, 640)
            t = torch.from_numpy(pred[None])
            out = unms.non_max_suppression(t, conf, 0.7, classes, False, max_det=300, nc=0)[0]
            out[:, :4] = ops.scale_boxes(inp, out[:, :4], image)
            mine = parity({"op": "nms", "pred": pred.ravel().tolist(), "nc": nc, "anchors": anchors, "conf": conf,
                           "classes": classes, "input": list(inp), "image": list(image)})
            theirs = [r[:5] + [int(r[5])] for r in out.tolist()]
            self.assertEqual(mine, theirs, f"trial {trial}")


def track_scenario(seed, frames=200):
    """Boxes of players skating about, crossing, disappearing and reappearing, with scores on
    both sides of the tracker's thresholds."""
    rng = np.random.default_rng(seed)
    objs = [{"x": rng.uniform(50, 590), "y": rng.uniform(40, 320), "vx": rng.normal(0, 3), "vy": rng.normal(0, 2),
             "w": rng.uniform(15, 50), "h": rng.uniform(40, 90), "cls": int(rng.choice([0, 1, 3, 4, 5]))}
            for _ in range(12)]
    out = []
    for _ in range(frames):
        dets = []
        for o in objs:
            o["x"] = float(np.clip(o["x"] + o["vx"] + rng.normal(0, 1), 0, 640))
            o["y"] = float(np.clip(o["y"] + o["vy"] + rng.normal(0, 1), 0, 360))
            if rng.random() < 0.02:
                o["vx"], o["vy"] = rng.normal(0, 4), rng.normal(0, 3)
            if rng.random() < 0.12:
                continue  # missed this frame
            j = rng.normal(0, 1.5, 4)
            x1, y1 = o["x"] - o["w"] / 2 + j[0], o["y"] - o["h"] / 2 + j[1]
            box = [max(0.0, x1), max(0.0, y1), min(640.0, x1 + o["w"] + j[2]), min(360.0, y1 + o["h"] + j[3])]
            dets.append(box + [float(rng.choice([rng.uniform(0.3, 0.95), rng.uniform(0.05, 0.3)], p=[0.8, 0.2])), o["cls"]])
        rng.shuffle(dets)
        out.append(np.asarray(dets, np.float32).reshape(-1, 6).tolist())
    return out


class ByteTrackTests(unittest.TestCase):
    def test_matches_ultralytics(self):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        for seed in range(4):
            frames = track_scenario(seed)
            tracker = BYTETracker(cfg)
            theirs = []
            for f in frames:
                t = tracker.update(Boxes(np.asarray(f, np.float32).reshape(-1, 6), (360, 640)))
                theirs.append([r[:4] + [int(r[4]), r[5], r[6], int(r[7])] for r in np.asarray(t, np.float32).reshape(-1, 8).tolist()])
            mine = parity({"op": "track", "frames": frames})
            for k, (a, b) in enumerate(zip(theirs, mine)):
                self.assertEqual(b, a, f"seed {seed}, frame {k}")


# ---- exact: jersey numbers and rosters

class JerseyTests(unittest.TestCase):
    def test_assemble_number_matches_run_models(self):
        import run_models
        rng = random.Random(0)
        cases = []
        for _ in range(500):
            digits = []
            for _ in range(rng.randint(0, 4)):
                x, y, h = rng.uniform(0, 80), rng.uniform(0, 80), rng.uniform(8, 20)
                w = h * rng.uniform(0.4, 0.8)
                digits.append({"class": str(rng.randint(0, 9)), "confidence": round(rng.uniform(0.25, 1), 4),
                               "box_xyxy": [round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)]})
            # sometimes a neighbouring second digit, so two-digit numbers come up
            if digits and rng.random() < 0.5:
                b = digits[0]["box_xyxy"]
                digits.append({"class": str(rng.randint(0, 9)), "confidence": round(rng.uniform(0.25, 1), 4),
                               "box_xyxy": [round(b[2] + 1, 1), b[1], round(b[2] + 1 + (b[2] - b[0]), 1), b[3]]})
            cases.append(digits)
        theirs = [list(r) if r else None for r in (run_models.assemble_number(c) for c in cases)]
        self.assertEqual(parity({"op": "assemble", "cases": cases}), theirs)

    def test_votes_match_run_models(self):
        import run_models
        rng = random.Random(1)
        steps, votes, theirs = [], run_models.JerseyVotes(), []
        for _ in range(3000):
            track = rng.randint(1, 6)
            kind = rng.choice(["add", "add", "add", "number", "settled"])
            if kind == "add":
                n = rng.choice(["12", "1", "2", "21", "7", "77", "9"])
                conf = round(rng.uniform(0.2, 1), 4)
                steps.append(["add", track, n, conf])
                votes.add(track, (n, conf))
            elif kind == "number":
                steps.append(["number", track])
                theirs.append(votes.number(track))
            else:
                steps.append(["settled", track])
                theirs.append(votes.settled(track))
        self.assertEqual(parity({"op": "votes", "steps": steps}), theirs)

    def test_roster_matches_roster_py(self):
        import roster
        r = roster.Roster(ROSTER, ("SJS", "MTL"))
        classes = ["team_a_player", "goalie_a", "team_b_player", "goalie_b", "referee", "other"]
        mine = parity({"op": "roster", "path": str(ROSTER), "teams": ["SJS", "MTL"]})
        self.assertEqual(mine["allowed"], [r.allowed(c) for c in classes])
        self.assertEqual(mine["players"], [list(r.player(c, str(n))) for c in classes for n in range(100)])
        self.assertEqual(mine["describe"], r.describe())


# ---- tolerance: the rink fit

class RinkFitTests(TempDir):
    def test_synthetic_clip_close_to_homography_py(self):
        clip = self.tmp / "syn"
        run_py(BENCH / "synth_detections.py", clip, "--frames", 600, "--seed", 1)
        run_py("homography.py", clip / "detections.json", clip / "placeholder.mp4")
        hv("homography", clip / "detections.json", "--frame-size", "1280x720", "--output", self.tmp / "rs")
        a = json.loads((clip / "positions.json").read_text())
        b = json.loads((self.tmp / "rs" / "positions.json").read_text())
        self.assertEqual(len(a), len(b))
        fit_diff = sum((x["homography"] is None) != (y["homography"] is None) for x, y in zip(a, b))
        # The Rust fit follows OpenCV 5's solver: ~0.2% of frames differ against OpenCV 5, up to
        # ~5% against OpenCV 4.x.
        self.assertLess(fit_diff / len(a), 0.06, f"{fit_diff} of {len(a)} frames differ in having a fit")
        d = sorted(math.dist(p["rink_xy"], q["rink_xy"]) for x, y in zip(a, b)
                   if sorted(x["keypoints_used"]) == sorted(y["keypoints_used"])
                   for p, q in zip(x["players"], y["players"]) if p["box_xyxy"] == q["box_xyxy"])
        self.assertGreater(len(d), 500)
        self.assertLess(d[len(d) // 2], 1e-4, "median player position difference, ft")


# ---- tolerance: the detection loop (needs the ONNX models and a video)

def need_models(test):
    missing = [p for p in [MODELS / "new_player_model.onnx", MODELS / "new_nums.onnx", VIDEO, ROSTER] if not p.exists()]
    return unittest.skipIf(bool(missing), f"needs {', '.join(map(str, missing))} (python rust/export_onnx.py)")(test)


class DetectTests(TempDir):
    FRAMES = 40

    @need_models
    def test_matches_run_models_on_the_same_onnx_models(self):
        """Rust vs run_models.py with the same ONNX models on the CPU."""
        weights = [f"{n}={MODELS / f}.onnx" for n, f in [("player", "new_player_model"), ("puck", "new_nano_puck"),
                                                         ("dots", "new_dots"), ("rink", "new_rink_model"),
                                                         ("number", "new_nums")]]
        common = ["--teams", "SJS", "MTL", "--roster", ROSTER, "--max-frames", self.FRAMES]
        run_py("run_models.py", VIDEO, "--device", "cpu", "--weights", *weights, *common, "--output", self.tmp / "py")
        hv("detect", VIDEO, "--device", "cpu", *common, "--output", self.tmp / "rs")
        from compare_detections import compare
        stem = VIDEO.stem
        a = json.loads((self.tmp / "py" / stem / "detections.json").read_text())
        b = json.loads((self.tmp / "rs" / stem / "detections.json").read_text())
        self.assertEqual(len(a), len(b))
        r = compare(a, b)
        # Only ONNX Runtime's float rounding differs (Rust bundles a newer ORT than Python's).
        self.assertGreaterEqual(r["identical"] / r["frames"], 0.8, "share of byte-identical frame records")
        self.assertEqual(r["cut_diff"], 0)
        for model, s in r["models"].items():
            self.assertEqual(s["a"], s["b"], f"{model}: box counts")
            self.assertEqual(s["matched"], s["a"], f"{model}: every box matched")
            self.assertLessEqual(r["worst"][model][0], 0.5, f"{model}: largest box difference, px")
        p = r["models"]["player"]
        self.assertEqual(r["ids_consistent"], r["ids_total"], "track ids")
        self.assertEqual(p["number_same"], p["matched"], "jersey numbers")

    @need_models
    def test_overlapped_and_sequential_runs_identical(self):
        common = ["--teams", "SJS", "MTL", "--roster", ROSTER, "--max-frames", 30, "--device", "cpu"]
        hv("detect", VIDEO, *common, "--output", self.tmp / "par")
        hv("detect", VIDEO, *common, "--sequential", "--output", self.tmp / "seq")
        stem = VIDEO.stem
        self.assertEqual((self.tmp / "par" / stem / "detections.json").read_bytes(),
                         (self.tmp / "seq" / stem / "detections.json").read_bytes())

    @need_models
    def test_run_writes_every_output_and_export_matches_python(self):
        hv("run", VIDEO, "--teams", "SJS", "MTL", "--roster", ROSTER, "--max-frames", 60, "--device", "cpu",
           "--output", self.tmp)
        clip = self.tmp / VIDEO.stem
        for f in ["detections.json", "positions.json", "positions.csv", "homographies.csv"]:
            self.assertTrue((clip / f).exists(), f)
        run_py("export_data.py", clip, "--output", self.tmp / "py_export")
        from compare_exports import differing_files
        self.assertEqual(differing_files(self.tmp / "py_export", clip / "export"), [])


if __name__ == "__main__":
    unittest.main()
