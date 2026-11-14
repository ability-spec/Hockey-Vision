# Hockey-Vision in Rust

The whole video-to-report pipeline in Rust, one stage per command and all of them in one
(`hockey-vision run`). [ANALYSIS.md](ANALYSIS.md) has the original analysis and plan. The Python
pipeline in the repo root is unchanged and stays the reference: each Rust stage reads and writes
the same files as the Python stage it replaces, so they can be swapped and compared one at a time.
Training, `export_engines.py`, `fetch_roster.py` and the web app stay in Python.

## What's here

| Stage | Python | Rust | vs Python (Apple M5, 600-frame 1080p60 clips) |
|---|---|---|---|
| Detection, tracking, numbers, cuts | `run_models.py`, `video_io.py`, `roster.py` | `hockey-vision detect` | **4.5x faster** with CoreML (50–54 vs 235 ms/frame); same output as Python's own ONNX run up to model float noise |
| Rink fit | `homography.py` | `hockey-vision homography` | ~20x faster on one core, ~90x on all; same fit on 96–100% of frames (see below) |
| Tables, events, report | `export_data.py`, `events.py` | `hockey-vision export` | 10–15x faster; **byte-identical** output |
| Whole pipeline | `jetson/run_clip.sh` | `hockey-vision run` | 37 s vs 178 s back to back (**4.8x**); 22–32 s on a quieter machine (up to 7x) |

```
rust/
├── crates/hv-core/      the pipeline without the models (no dependencies beyond serde/nalgebra/rayon)
│   └── src/
│       ├── image.rs     decoding colour conversion, resizes, letterbox, cut detector (OpenCV-exact)
│       ├── yolo.rs      Ultralytics NMS and box scaling
│       ├── bytetrack.rs Ultralytics' BYTETracker, Kalman filter, assignment
│       ├── jersey.rs    assemble_number, JerseyVotes (run_models.py)
│       ├── roster.rs    roster.py
│       ├── homography.rs, geometry.rs, rink.rs, pipeline.rs   homography.py
│       ├── export.rs, events.rs, columns.rs                   export_data.py, events.py
│       └── pyfmt.rs, pyval.rs   Python's float repr, round, hypot, json.dumps, csv quoting
├── crates/hv-infer/     ONNX Runtime models, ffmpeg frame reader, the detect loop
├── crates/hv-cli/       the `hockey-vision` binary
├── export_onnx.py       exports the five models to CV_Models/Models/onnx/
└── bench/               parity and timing scripts against the Python pipeline
```

## Setup

Needs Rust 1.81 or newer ([rustup.rs](https://rustup.rs)), `ffmpeg`/`ffprobe` on the PATH, and the
models as ONNX (once, with the Python environment):

```bash
python rust/export_onnx.py                          # -> CV_Models/Models/onnx/*.onnx
cd rust
cargo build --release --features coreml             # on a Mac; leave out --features coreml elsewhere
cargo test --release
```

The first build downloads ONNX Runtime (via the `ort` crate).

## Run

```bash
B=rust/target/release/hockey-vision
$B run videos/<clip>.mp4 --teams SJS MTL --device coreml --time   # everything, like jetson/run_clip.sh
$B detect videos/<clip>.mp4 --teams SJS MTL                         # run_models.py
$B homography outputs/<clip>/detections.json --frame-size 1920x1080 # homography.py
$B export outputs/<clip>                                            # export_data.py
```

`detect` and `run` take run_models.py's options (`--models`, `--conf`, `--puck-conf`, `--teams`,
`--roster`, `--jersey-stride`, `--rink-stride`, `--max-frames`, `--output`, `--weights NAME=PATH`)
plus `--device cpu|coreml`, `--threads N` (default: performance cores), `--models-dir`,
`--sequential` (no overlapping; same output) and `--time` (per-stage timings).

Not ported: images (only videos), `--width`, `--save-video` / `--per-model` / `--debug` videos, and
TensorRT engines. Use the Python scripts for those.

## How close it is to Python

**Export** (`bench/compare_exports.py`): byte-identical on the three real clips, a multi-clip export
and three synthetic 30,000-frame games (`bench/synth_game.py`: 1,469 events, ~1.1M table rows).
This needs CPython 3.9's exact `round`, `math.hypot` and float formatting, which `pyval.rs`
reproduces (checked bit for bit on 1M values by `cargo run --example check_pymath`). Python 3.10+
changed `math.hypot`'s last-bit rounding, so against a 3.10 Python a distance can differ in the
last digit.

**Detection** (`bench/compare_detections.py`). Every pixel step matches OpenCV exactly
(`bench/check_pixels.py`: decoding, the 1/3 downscale, 300 crop resizes, HSV, 180M values, 0
different). Against `run_models.py` with the same ONNX models on the CPU:

| Clip | Identical frame records | Boxes identical | Track ids | Jersey numbers | Cuts |
|---|---|---|---|---|---|
| 02:48 | 96.2% | 10,362 / 10,386 | 100% | 100% | same |
| 20:48 | 96.8% | 10,669 / 10,690 | 100% | 100% | same |
| 47:04 | 96.3% | 11,394 / 11,416 | 100% | 100% | same |

The boxes that differ are 0.1 px / 0.0001 confidence apart, with one digit box 0.4 px. That is
ONNX Runtime itself: Rust bundles ORT 1.22, Python has 1.19, and on the same input tensor their raw
outputs agree on only ~46% of values (up to 7e-4 apart). CoreML against Python's default
(`.pt` on the GPU) is just as close (94–96% identical records, 100% track ids and numbers), the
same spread as Python's own ONNX run vs its `.pt` run.

**Rink fit** (`bench/compare_positions.py`): the Rust fit reproduces OpenCV 5's solver. Against
`homography.py` on OpenCV 5, with the same detections: clips 20:48 and 47:04 match to 0.01 ft on
every frame; 02:48 differs on 1 frame of 600 (max 0.69 ft elsewhere); a synthetic 3,000-frame clip
on 0.2% of frames. Against OpenCV 4.11 (this repo's venv) it is 0–2 frames per clip, up to 27 ft.
The venv can't move to OpenCV 5 yet: it needs numpy 2, and Ultralytics on macOS excludes numpy
2.0–2.3.4, which is all Python 3.9 can install. A separate venv with only opencv-python 5 runs
homography.py for comparisons.

## Speed (Apple M5, 600-frame 1080p60 clips)

| | Python | Rust |
|---|---|---|
| Detection, default device | 235 ms/frame (`.pt` on MPS) | 50–54 ms/frame (CoreML) |
| Detection, CPU / ONNX | ~500 ms/frame (Ultralytics + onnxruntime) | 300–330 ms/frame |
| Rink fit | 2–3 ms/frame | 0.1 ms/frame |
| Export | 0.2 ms/frame | 0.02 ms/frame |

Where detection time goes with CoreML: four full-frame models at ~7 ms each run at the same time,
the number model reads ~14 player crops one at a time (~4 ms each), and frame N+1's models run while
frame N's numbers are read. CoreML is fast only with fixed input shapes, so each model gets a
fixed-shape session for the shapes it sees every frame (the 640x384 frame, a 640x640 crop) and
anything else falls back to the CPU session. Overlapping doesn't change the output (checked byte
for byte).

## Tests

```bash
(cd rust && cargo test --release)                                   # Rust unit tests
python -m unittest discover -s tests -p "test_rust_port.py" -v      # Rust vs the Python pipeline (~70 s)
```

`tests/test_rust_port.py` builds the binary and checks each Rust component against the Python code
it replaces:

| Exact (bit for bit) | Within tolerance |
|---|---|
| export_data.py on synthetic games, several clips, no positions | rink fit vs homography.py (fit on all but <6% of frames, median position difference < 0.0001 ft) |
| Ultralytics NMS + box scaling, 30 random cases | detection vs run_models.py on the same ONNX models (≥80% identical records; every box matched within 0.5 px; same track ids, numbers, cuts) |
| Ultralytics BYTETracker, 4 x 200 frames of crossing, vanishing players | |
| assemble_number, JerseyVotes, roster.py | |
| CPython's hypot / divmod (100k values) | |
| OpenCV decoding, resizes, cut histograms (generated video) | |
| overlapped vs sequential detection; `run`'s export vs export_data.py | |

The detection tests are skipped without the ONNX models (`python rust/export_onnx.py`) and a video
(`HV_TEST_VIDEO`, default outputs/speedup_test/clips/window1_02m48s.mp4).

## Parity and timing scripts

| Script | Does |
|---|---|
| `bench/compare_detections.py` | detections.json vs detections.json: matched boxes, track ids, numbers, cuts |
| `bench/compare_positions.py` | positions.json vs positions.json |
| `bench/compare_exports.py` | export folder vs export folder, byte for byte |
| `bench/check_pixels.py` | Rust decoding / resizing / histograms vs OpenCV on real frames |
| `bench/synth_game.py` | synthetic detections + positions with thousands of events |
| `bench/synth_detections.py`, `run_parity.sh`, `run_python_homography.py` | rink-fit parity and timing |
| `bench/gen_columns.sh` | regenerates `columns.rs` from export_data.py |
| `cargo run --release --example bench_model` | time one ONNX model on CPU or CoreML |
