# Running on a Jetson Orin Nano

Setup: JetPack 6 on the Orin Nano, everything run inside Ultralytics' Jetson Docker image
(CUDA PyTorch from `pip` doesn't exist for the Jetson, so a plain `pip install -r requirements.txt`
would run on the CPU).

## 1. One-time setup on the Jetson

```bash
sudo jetson_clocks                      # run the clocks at max (resets on reboot)
git clone https://github.com/ability-spec/Hockey-Vision.git
cd Hockey-Vision
sudo docker build -t hockey-vision jetson/
```

Put the image and the outputs on the NVMe drive if the Jetson has one; the image is several GB.

## 2. Copy the models and videos over (from your Mac)

`CV_Models/` and `videos/` aren't in git.

```bash
scp -r CV_Models videos <user>@<jetson-ip>:~/Hockey-Vision/
```

## 3. Start the container

```bash
cd ~/Hockey-Vision
sudo docker run -it --rm --runtime=nvidia --ipc=host -v "$PWD":/work hockey-vision bash
```

The repo is mounted at `/work`, so everything written there lands in `~/Hockey-Vision` on the
Jetson and survives the container.

## 4. Build the TensorRT engines (once, inside the container)

```bash
python export_engines.py
```

This writes `CV_Models/<model>_model.engine` for each model (FP16), which `run_models.py` then
uses automatically. It takes a while. If a build runs out of memory, add swap on the Jetson
and retry just that model with `--models <name>`.

## 5. Run a clip

```bash
jetson/run_clip.sh videos/<clip>.mp4                    # whole video
jetson/run_clip.sh videos/<clip>.mp4 --max-frames 1800  # first 1800 frames only
```

This skips all annotated videos and writes, in `outputs/<clip>/`:

| File | Contents |
|---|---|
| `homographies.csv` | one row per frame: `frame, time_s, rejected, keypoints_used, h00 … h22` (empty when no fit) |
| `positions.csv` | one row per player / puck per frame: `frame, time_s, object, class, jersey_number, confidence, x_ft, y_ft` |
| `positions.json` | the same, nested per frame |
| `detections.json` | the raw model detections (input to `homography.py`) |

To also get `side_by_side.mp4`, run `homography.py` without `--no-video`.

## 6. Copy the results back (from your Mac)

```bash
scp <user>@<jetson-ip>:~/Hockey-Vision/outputs/<clip>/{homographies.csv,positions.csv,positions.json} .
```

## Using the homography

Each row of `homographies.csv` is the 3x3 matrix mapping a video pixel `(u, v)` to rink feet
(origin at centre ice, x -100..100 along the rink, y -42.5 near boards .. 42.5 far boards):

```python
import numpy as np, pandas as pd

H = pd.read_csv("homographies.csv").dropna(subset=["h00"])
row = H.iloc[0]
M = row[[f"h{r}{c}" for r in range(3) for c in range(3)]].to_numpy(float).reshape(3, 3)
x, y, w = M @ [960, 700, 1]
print(row.frame, x / w, y / w)   # rink feet of pixel (960, 700) in that frame
```

