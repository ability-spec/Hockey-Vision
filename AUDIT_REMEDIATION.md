# October audit remediation

| Finding | Change |
|---|---|
| H01 | Propagate `--output DIR` and `--output=DIR` to the homography input path, including paths with spaces. |
| H02 | Prefer TensorRT engines only for CUDA devices; CPU/MPS use .pt weights, including per-video player reloads. |
| H03 | Raise on videos that cannot open or decode any frames. The wrapper stops on failure and refuses stale detections. |
| H04 | Jetson instructions clone ability-spec/Hockey-Vision. |

## Verification

**11 CPU-only tests passed** using model/capture doubles and real shell subprocesses. These test actual extracted CLI functions and wrapper argument forwarding, not detector accuracy.

```bash
python -m pip install pytest
python -m pytest tests/test_pipeline_cli.py -q
```

No Jetson, TensorRT engine, model weights or annotated hockey footage was evaluated. Existing TensorRT engines must still match the CUDA/TensorRT hardware/runtime; file presence is not a compatibility guarantee. Detection freshness uses file timestamps, so the wrapper may fail closed on coarse or unusual filesystems. A partially corrupt video after its first decoded frame is not diagnosed by this change.
