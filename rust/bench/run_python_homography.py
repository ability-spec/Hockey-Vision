"""Run homography.py's main() with one guard added: a singular fit counts as no fit.

homography.py has this guard itself now; this wrapper is kept for running older checkouts.

cv2.findHomography can return a singular matrix while compute_homography() tries line diagonals,
and _refine_circle_edges() then raises numpy's "Singular matrix", ending the run. The Rust port
(rust/crates/hv-core/src/homography.rs, fit()) treats such a fit as no fit; this applies the same
rule to the Python baseline, so both run the whole clip and their outputs can be compared.
Everything else is homography.py unchanged.

    python rust/bench/run_python_homography.py <detections.json> <video> [homography.py options]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import homography  # noqa: E402

_fit = homography._fit


def _guarded_fit(matches):
    H, err = _fit(matches)
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        return None, None
    return H, err


homography._fit = _guarded_fit

if __name__ == "__main__":
    sys.argv[0] = homography.__file__
    homography.main()
