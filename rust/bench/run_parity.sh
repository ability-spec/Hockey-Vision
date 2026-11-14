#!/usr/bin/env bash
# Build the Rust port, run it and homography.py on the same synthetic clip, time both and compare
# their positions.json. Needs cargo, numpy and opencv-python (homography.py's own dependencies).
#
#   rust/bench/run_parity.sh [frames] [seed]        # from the repo root; default 3000 frames, seed 0
set -euo pipefail

frames=${1:-3000}
seed=${2:-0}
root=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

(cd "$root/rust" && cargo build --release -q)
bin="$root/rust/target/release/hockey-vision"

python "$root/rust/bench/synth_detections.py" "$work/py" --frames "$frames" --seed "$seed"
mkdir -p "$work/rs"

echo "== homography.py (with the singular-fit guard, see run_python_homography.py)"
start=$(date +%s.%N)
python "$root/rust/bench/run_python_homography.py" "$work/py/detections.json" "$work/py/placeholder.mp4" > "$work/py.log"
grep "Rink fit" "$work/py.log"
py_s=$(echo "$(date +%s.%N) - $start" | bc)

echo "== Rust, 1 thread"
"$bin" homography "$work/py/detections.json" --frame-size 1280x720 --output "$work/rs" --sequential --time | tail -1
echo "== Rust, all cores"
"$bin" homography "$work/py/detections.json" --frame-size 1280x720 --output "$work/rs" --time | sed -n '1p;3p'

printf "== homography.py took %.1f s (%.2f ms/frame, including start-up)\n" "$py_s" "$(echo "$py_s * 1000 / $frames" | bc -l)"
echo "== parity"
python "$root/rust/bench/compare_positions.py" "$work/py/positions.json" "$work/rs/positions.json"
