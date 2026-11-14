"""Export the five YOLO models to ONNX for the Rust pipeline (hockey-vision detect / run).

    python rust/export_onnx.py            # -> CV_Models/Models/onnx/<model>.onnx

Dynamic shapes, FP32, with Ultralytics' metadata (class names, stride), which hockey-vision reads.
The Rust code pins the shapes itself where that is faster (CoreML).
"""
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parents[1] / "CV_Models" / "Models"
MODEL_FILES = {
    "player": "new_player_model",
    "puck": "new_nano_puck",
    "dots": "new_dots",
    "rink": "new_rink_model",
    "number": "new_nums",
}
NUMBER_BATCH = 16  # the number model reads player crops in batches of this size

OUT = MODELS_DIR / "onnx"
OUT.mkdir(parents=True, exist_ok=True)
for name, stem in MODEL_FILES.items():
    tmp = Path(tempfile.mkdtemp())
    weights = Path(shutil.copy(MODELS_DIR / f"{stem}.pt", tmp / f"{stem}.pt"))
    path = Path(YOLO(str(weights)).export(format="onnx", imgsz=640, dynamic=True, simplify=True,
                                          batch=NUMBER_BATCH if name == "number" else 1, verbose=False))
    shutil.move(str(path), OUT / f"{stem}.onnx")
    print(f"{name}: {OUT / f'{stem}.onnx'}")
