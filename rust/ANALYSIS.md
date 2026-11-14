# Rust for Hockey-Vision: analysis

> **Update (Nov 2026):** phases 1 and 2 are done for the Mac, and export is ported too. Every
> stage now runs in Rust (`hockey-vision run`), 4.5–5.3x faster end to end on an M5 with CoreML,
> with the same outputs as Python (byte-identical tables and events; detections equal up to ONNX
> Runtime float noise). See [README.md](README.md) for the measurements. TensorRT / Jetson support
> (hv-infer with the TensorRT execution provider or engines) is still to do.

What it would take to run Hockey-Vision in Rust, what it would gain, and in what order to do it.
Every number here was measured for this report unless it is marked as an estimate. Section 7 has
the commands to reproduce them.

## 1. Summary

- **The models won't get faster.** Rust would run the same TensorRT engines on the same GPU. What
  Rust removes is the work around the models: Python pre- and post-processing per model call,
  number-crop preprocessing on the CPU, the GIL that keeps decoding, inference and writing from
  running in parallel, and the JSON round trips between stages.
- **`homography.py` is the clearest win, and it is now ported** (`rust/crates/hv-core`). On
  synthetic broadcast clips it runs **~22x faster on one core and 75-88x on four**, and it gives the
  same answer: 99.7-99.8% of frames get the same fit, and on those frames the player positions
  match to 0.00000 ft at the 99th percentile. Python takes 20-77 ms per frame for this stage,
  which comes to 35-140 minutes for an hour of 30 fps video. That is comparable to, or slower
  than, the inference stage it follows.
- **In `run_models.py`, the Python cost is mostly preprocessing.** Per frame on an x86 core: about
  10 ms of CPU letterboxing for the number model's crops (0.72 ms per crop, 14 players), about
  1.5 ms of Ultralytics pre- and post-processing for each of the four full-frame models, and 3 ms
  of camera-cut detection. That is roughly 19 ms per frame before any GPU work. On the Orin Nano's
  slower cores this is *estimated* at 40-60 ms per frame, all of it serial with the GPU work. A
  Rust runner that preprocesses on the GPU and overlaps stages could remove most of it. That gain
  has to be confirmed on the Jetson.
- **The export and events stage isn't worth porting yet.** It runs at 0.18 ms per frame (about
  1 minute for a full game) and is pure standard-library Python that the web app also uses.
- **Recommendation: port incrementally, and keep Python for training, export, the web app and
  rosters.** Phase 1 is done. Phase 2 is a Rust inference runner (TensorRT, GPU preprocessing,
  ByteTrack, jersey votes) that writes the same `detections.json`. Phase 3 overlaps the stages
  and decodes zero-copy. See section 5.

## 2. Where the time goes today

The pipeline is three programs run one after another (`jetson/run_clip.sh`):

```
video ─► run_models.py ─► detections.json ─► homography.py ─► positions.json ─► export_data.py ─► export/
         decode, 5 YOLO26n models,            rink fit per frame,                tables, events,
         ByteTrack, jersey votes              smoothing, projection              game_report.json
```

### 2.1 `run_models.py` (detection, tracking, numbers)

Per frame, the main thread does all of the following, in order:

1. Camera-cut detection: quarter-size resize, HSV conversion, 2D histogram.
2. A downscale to the models' input size, shared by the full-frame models.
3. Four `model.predict()` / `model.track()` calls (player with ByteTrack, puck, rink, dots), one
   after another. Each does its own Ultralytics preprocessing, a forward pass and postprocessing,
   then `.cpu()`, which waits for the GPU.
4. The number model: player crops cut from the full-resolution frame. Ultralytics letterboxes each
   crop to 640 on the CPU, then runs a batch of up to 16.
5. Python bookkeeping: digit filtering, `assemble_number`, `JerseyVotes`, `results_to_dicts`,
   `json.dumps`. A background thread draws and writes the outputs, and another decodes frames.

Measured here (Intel Xeon @ 2.8 GHz, 3 torch threads, `rust/bench/ultralytics_overhead.py`, YOLO26n
with random weights, which costs the same as the trained models):

| Per frame | ms | Notes |
|---|---|---|
| Full-frame `predict()`: Ultralytics preprocess + postprocess | 0.86 + 0.60 | per model, ×4 models. Random weights give few boxes, so postprocess and ByteTrack are lower bounds |
| Number model preprocessing | 0.72 per crop, **~10 for 14 players** | CPU letterbox of each crop to 640, then stacking and normalising |
| Camera-cut detection, 1080p | 3.0 | `CutDetector.update` |
| Downscale 1080p → 640×360 | 0.2 | |
| `results_to_dicts` + `json.dumps` | 0.16 | |
| Jersey votes (14 tracks) | 0.05 | |
| **Total CPU work around the models** | **≈ 19** | serial with the GPU work: each call waits for its result |

On the Jetson Orin Nano (6 × Cortex-A78AE at 1.5-1.7 GHz), single-core speed is *estimated* at 2-3x
below this Xeon's, which would put the same work at **~40-60 ms per frame**, before any time on
the GPU. We have no GPU timings for the Jetson. Run
`python rust/bench/ultralytics_overhead.py --device cuda` there (it needs no weights) to get
the real split between CPU overhead and GPU time. That number decides how much phase 2 is worth.

The bookkeeping that is often blamed on Python (dicts, JSON, votes) is under 0.3 ms per frame.
The cost is in preprocessing and in serialisation: nothing overlaps the GPU's work.

### 2.2 `homography.py` (rink fit)

This stage needs no models, so it was timed directly on synthetic clips
(`rust/bench/synth_detections.py`: a pinhole broadcast camera that pans and zooms, with rink
markings, players and puck projected into boxes with pixel noise):

| Clip | Python (ms/frame) | Rust, 1 thread | Rust, 4 threads | Speed-up 1 / 4 threads |
|---|---|---|---|---|
| 3,000 frames, seed 0 | 50.9 | 2.33 | 0.61 | 22x / 84x |
| 600 frames, seed 1 | 19.5 | 0.90 | 0.22 | 22x / 88x |
| 600 frames, seed 3 | 77.0* | 3.50 | 1.03 | 22x / 75x |

\* includes about 1 s of Python start-up (0.8 s of it imports and the template).

A cProfile run shows where Python's time goes. Each frame makes about 45 `cv2.findHomography`
calls, each followed by circle-edge refinement (`_refine_circle_edges`: inverting H and
reprojecting 180 ring points four times). That's because when the point landmarks alone can't fit,
`compute_homography` tries **every combination of line-box diagonals** (2^n for n line boxes; the
synthetic frames have 3-5 of them, so 8-32 robust fits per frame). Each call works on arrays of
5-15 points, so numpy and OpenCV spend more time crossing between Python and C than computing.
This is the kind of work where Rust gains most.

The synthetic clips probably have more hard frames than real broadcasts do (their fit rate is low:
15-62%), and the cost per frame grows with the number of line boxes. Real clips may be faster
in both languages. The ratio should hold, because the work per fit is the same.

### 2.3 `export_data.py` / `events.py`

On a 30,000-frame clip (rink positions written by the Rust port, read by `export_data.py`
unchanged), the export took **5.3 s, or 0.18 ms per frame**, and scaled linearly from 3,000
frames (0.65 s). A full game is about a minute. Porting it would save a minute per game and
duplicate the code the web app shares. It should be the last thing ported.

### 2.4 Between the stages

`detections.json` is about 1.6 KB per frame as text: 49 MB for 30,000 frames, and about 500 MB for
a full game. `homography.py` and `export_data.py` both `json.loads` the whole file at once, which
needs several times its size in memory, and the Jetson has 8 GB. A single Rust binary can pass
detections between stages in memory and fit the homography as frames come out of the detector,
keeping the files as outputs only.

## 3. What Rust changes, and what it doesn't

| Cost | Rust helps? | How |
|---|---|---|
| TensorRT kernels | No | Same engines, same GPU. INT8/FP16 choices stay what they are |
| Ultralytics preprocessing (letterbox, normalise, HWC→CHW) | **Yes**, mostly by moving it to the GPU | One CUDA kernel letterboxes the frame for all four full-frame models, and a batched crop-resize kernel prepares the number model's crops straight from the decoded frame on the GPU. No 6 MB-per-frame CPU copies |
| Postprocessing (YOLO26 is NMS-free) | Yes | Thresholding a `[N, 300, 6]` output is microseconds in Rust. INT8 engines on TensorRT 10.3 lack the NMS-free head and need a small NMS, which is also cheap |
| Waiting for each model before starting the next | **Yes** | The puck, rink and dots models don't depend on the player model: put them on separate CUDA streams. The number model only waits for player boxes |
| Decode, infer and write fighting over the GIL | **Yes** | Real threads: decode frame N+2, infer N+1 and postprocess N at the same time. GStreamer `nvv4l2decoder` into NVMM memory gives the GPU the frame without a CPU copy |
| ByteTrack, jersey votes, JSON | Yes, though they're small | A port of Ultralytics' BYTETracker with the same `bytetrack.yaml` settings (high 0.25, low 0.1, new 0.25, buffer 30, match 0.8, fuse_score) |
| `homography.py` | **Yes, 22x on one core**, and fits are per frame, so they parallelise | Done in phase 1 |
| Stages that write and re-read JSON | Yes | One binary, detections passed in memory |
| Memory | Yes | Streaming instead of loading whole JSON files |

What Rust doesn't get on its own: some of these gains are possible in Python too. TensorRT's
Python API, CUDA streams and GPU letterboxing with torch would remove much of the preprocessing
cost without leaving Python. What only Rust (or C++) gives is the rest together: no GIL, so the
stages really overlap; low overhead per call on the Jetson's slow cores; predictable latency;
and a single static binary in the container. Rust has C++'s speed with memory safety and a
manageable build.

## 4. Proposed architecture

```
rust/
├── crates/hv-core     geometry, rink, homography fit, projection, jersey votes     (phase 1: done)
├── crates/hv-cli      `hockey-vision` binary                                       (phase 1: `homography`)
├── crates/hv-infer    TensorRT engines, GPU pre/postprocessing, CUDA streams        (phase 2)
├── crates/hv-track    ByteTrack, Ultralytics-compatible                             (phase 2)
├── crates/hv-video    decoding: GStreamer/NVDEC on the Jetson, FFmpeg elsewhere     (phase 2-3)
└── bench/             parity and timing scripts against the Python pipeline
```

Library choices:

| Need | Choice | Why / alternatives |
|---|---|---|
| Inference on the Jetson | **TensorRT runtime directly** (C++ API via `cxx`, or the `tensorrt`/`trt` crates) | Loads the `.engine` files `export_engines.py` already builds. Ultralytics prepends a 4-byte length and a JSON metadata block (what `engine_imgsz()` reads), which has to be skipped before deserialising. Keeps FP16/INT8 engines and the 384×640 input |
| Inference on Mac / desktop | **`ort`** (ONNX Runtime) with the CoreML, CUDA or TensorRT execution provider | Needs ONNX exports (`export_engines.py` could add `format="onnx"`). `ort` can also run on the Jetson with NVIDIA's ONNX Runtime builds, as a fallback |
| Not recommended | `candle`, `burn`, `tract` | No TensorRT, so much slower on the Jetson; `tract` is CPU-only |
| GPU kernels (letterbox, crop-resize, normalise) | `cudarc` with small NVRTC kernels, or NPP | One kernel launch per frame for all full-frame inputs, one for all crops |
| Decoding | `gstreamer-rs` (`nvv4l2decoder`, NVMM) on the Jetson; `ffmpeg-next` elsewhere | Same choice `video_io.py` makes today |
| Linear assignment (ByteTrack) | Port of LAPJV (`lap` in Python), or the `lapjv` crate | Small matrices, so the choice is about matching Python's track ids |
| JSON, CSV | `serde_json`, plain writers in Python's formatting (`hv_core::pyfmt`) | Outputs stay byte-for-byte diffable against Python's |
| Parallel CPU work | `rayon` | Already used for per-frame fits |
| Python interop (optional) | `pyo3` + `maturin` | Lets `homography.py` call `hv-core` directly, a low-risk way to get phase 1's speed inside the existing Python flow |

Python stays for training the models, `export_engines.py`, `fetch_roster.py`, the web app, and
`export_data.py` / `events.py` until there is a reason to move them. The file formats are the
interface, so each Rust stage can replace its Python counterpart one at a time and be checked
against it.

## 5. Plan

| Phase | Scope | Expected gain | How it is validated |
|---|---|---|---|
| **1 (this PR)** | `hv-core`: rink keypoints, OpenCV-compatible `findHomography` (normalised DLT plus OpenCV 5's LevMarq), keypoint matching, robust fitting, line diagonals, tracker, projection, rejection, output writers. `hockey-vision homography` replaces `homography.py` when no videos are wanted. Jersey votes ported for phase 2 | Measured: 22x on one core, 75-88x on four, for the rink-fit stage | `bench/run_parity.sh`: 99.7-99.8% of frames identical, the rest borderline fits (section 6) |
| 2 | `hv-infer` + `hv-track` + `hv-video`: `hockey-vision detect video.mp4` writes the same `detections.json` as `run_models.py`, with TensorRT, GPU preprocessing, ByteTrack and jersey votes | *Estimate:* removes the ~19 ms/frame (x86) or ~40-60 ms/frame (Jetson) of CPU work around the models | On the `CV_Models/Examples` clips: box IoU against `run_models.py`, track-id agreement, identical jersey numbers per track |
| 3 | Overlap: CUDA streams for the four full-frame models, a decode / infer / postprocess pipeline, NVMM zero-copy decode | *Estimate:* frame time drops to about max(GPU time, decode time) instead of their sum | Frames per second on the Jetson, and outputs identical to phase 2 |
| 4 | `hockey-vision run`: detection and rink fit in one pass, no intermediate JSON, streaming outputs | No re-parsing, flat memory use on full games | Outputs identical to phases 2 + 1 run separately |
| 5 (optional) | Port `events.py` / `export_data.py`, PyO3 bindings for the web app | About a minute per game | The existing `tests/test_events.py` cases ported, plus diffs of the exports |

What phase 2 would need:

- **Weights and example clips** for parity runs. `run_models.py`'s outputs on the example clips
  can be the test fixtures.
- **On-device timings first.** `rust/bench/ultralytics_overhead.py --device cuda` on the Jetson
  shows how much of each frame is CPU overhead, and so how much phase 2 can gain, before any
  inference code is written.
- **A Rust toolchain in the Jetson image** (`rustup` in `jetson/Dockerfile`, building against
  JetPack 6's TensorRT 10.3 and CUDA 12.x), or cross-compiled aarch64 binaries.

## 6. Parity details and risks

**Phase 1 parity.** `hv-core` mirrors `homography.py` function by function: same names, same
order of operations, same thresholds. It also mirrors numpy's `float32` rounding where
Python has it. The fit itself is a port of what `cv2.findHomography(src, dst, 0)` does in OpenCV
5: the normalised DLT (`runKernel`), then `cv::LevMarq` over all nine entries of H with 10
iterations and geodesic acceleration. On a noisy test case it agrees with cv2 to one float32 ulp
(`geometry::opencv_parity`). An earlier version used the OpenCV 4 solver or ran LM to full
convergence, and they differed on 12% and 9% of frames respectively, because the robust fit amplifies
small differences. The optimiser has to match OpenCV's exactly, not just be good.
Parity was measured against OpenCV 5.0, which `pip install opencv-python` installs today.
OpenCV 4.x (JetPack's `python3-opencv`, for example) refines with the older `LMSolver` and so gives
slightly different fits itself, on about the same share of frames. Both are valid fits. The Rust
port follows OpenCV 5.

What still differs (0.17-0.33% of frames): frames where the outlier-removal loop passes through
a garbage fit (15-40 ft errors). Starting from there, circle-edge refinement is chaotic, and
ulp-level differences between OpenCV's Jacobi eigen/SVD and nalgebra's send the two
implementations to different endings. In each such frame, at least one of them ends with a
borderline fit or none. A frame whose fit differs can also carry a small difference into the
next few frames through the tracker's smoothing (worst seen: 1.2 ft, for one player in one
frame). Porting OpenCV's Jacobi routines would close most of this gap. These frames are
borderline in Python too, so this didn't seem worth it.

**Deliberate difference.** A fit whose matrix is singular counts as no fit. In Python,
`np.linalg.inv` raises on it inside `_refine_circle_edges` and **`homography.py` aborts the
whole clip**: it crashed at about frame 350 of the first synthetic clip. That is a bug in the
current Python code, independent of Rust (see `bench/run_python_homography.py`, which adds the same
guard to the Python baseline for the comparisons).

**Risks:**

- *Two implementations to maintain.* A change to a threshold in Python has to be made in Rust
  too. Keeping the Rust functions named and ordered like the Python ones, plus
  `bench/run_parity.sh`, keeps this manageable. Once a stage is ported, its Python version
  can be kept as the reference or retired.
- *TensorRT and JetPack coupling.* Engines are tied to the TensorRT version, and Ultralytics'
  engine header and YOLO26 output layout may change between versions. Phase 2 should pin both
  and test the header parsing.
- *Tracker parity.* Track ids drive jersey votes and every per-player number downstream. The
  ByteTrack port has to match Ultralytics' exactly (Kalman filter, `fuse_score`, LAPJV tie-breaking)
  or the outputs will drift.
- *Build complexity on the Jetson.* Bindings to TensorRT and GStreamer need the JetPack
  libraries at build time. Do the build inside the existing Ultralytics Jetson image.

## 7. Reproducing the numbers

From the repo root (needs `cargo`, `numpy`, `opencv-python`; the Ultralytics overhead script
also needs `torch` and `ultralytics`):

```bash
rust/bench/run_parity.sh 3000 0                       # build, time both rink fits, compare outputs
python rust/bench/ultralytics_overhead.py --device cpu  # or --device cuda on the Jetson
(cd rust && cargo test --release)                     # Rust unit tests, incl. OpenCV parity
```

Other tools in `rust/bench/`: `synth_detections.py` writes synthetic `detections.json`;
`compare_positions.py` diffs two `positions.json` files; `dump_fits.py` (with
`cargo run --release --example dump_fits`) dumps each frame's raw fit from both
implementations, to find where they first part ways.

## 8. Found along the way

- `homography.py` crashes with `LinAlgError: Singular matrix` when `cv2.findHomography` returns
  a singular H during the line-diagonal search. It loses the whole clip's output, because
  `positions.json` is only written at the end. The fix is a determinant check in `_fit()`,
  filed as a separate task.
- In either language, the 2^n line-diagonal search is most of `homography.py`'s cost. Pruning it
  (for example, fitting each line's diagonal on its own against the point fit, or stopping once
  a combination has a low error) would speed up Python and Rust alike.
- Number-crop preprocessing is the largest Python cost in `run_models.py`. Even without Rust,
  letterboxing the crops on the GPU (torch ops on the GPU frame) instead of in Ultralytics' CPU
  preprocessing would help.
