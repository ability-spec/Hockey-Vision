"""Compare two positions.json files frame by frame: homography.py's and the Rust port's.

    python rust/bench/compare_positions.py python/positions.json rust/positions.json

Reports how many frames agree on whether there is a fit, the keypoints it used, the camera view
and why a fit was rejected, and how far apart the player and puck rink positions are on frames
whose fits agree. Exits non-zero when more than --max-mismatch of the frames differ in their fit
or the 99th percentile of position differences on agreeing frames exceeds --max-ft (a frame
whose fit differs can carry a small difference into the next few through the tracker's smoothing). A frame both reject counts as
agreeing even if the reasons differ (the output is the same: no fit). Standard library only.
"""

import argparse
import json
import math
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a", help="positions.json (e.g. Python's)")
    parser.add_argument("b", help="positions.json (e.g. Rust's)")
    parser.add_argument("--max-mismatch", type=float, default=0.005,
                        help="fraction of frames allowed to differ in fit / rejection / keypoints (default 0.005)")
    parser.add_argument("--max-ft", type=float, default=0.05,
                        help="largest allowed 99th-percentile position difference where both agree, ft (default 0.05)")
    args = parser.parse_args()
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    if len(a) != len(b):
        sys.exit(f"different frame counts: {len(a)} vs {len(b)}")

    n = len(a)
    fit_diff, rejected_diff, view_diff, kp_diff = [], [], [], []
    player_d, puck_d, player_count_diff, worst_frame = [], [], 0, None
    for fa, fb in zip(a, b):
        if (fa["homography"] is None) != (fb["homography"] is None):
            fit_diff.append(fa["frame"])
        if fa["rejected"] != fb["rejected"]:
            rejected_diff.append(fa["frame"])
        if fa["view"] != fb["view"]:
            view_diff.append(fa["frame"])
        if sorted(fa["keypoints_used"]) != sorted(fb["keypoints_used"]):
            kp_diff.append(fa["frame"])
            continue  # different fits: counted above, positions not comparable
        pa = {tuple(p["box_xyxy"]): p["rink_xy"] for p in fa["players"]}
        pb = {tuple(p["box_xyxy"]): p["rink_xy"] for p in fb["players"]}
        player_count_diff += pa.keys() != pb.keys()
        for k in pa.keys() & pb.keys():
            d = math.dist(pa[k], pb[k])
            if not player_d or d > max(player_d):
                worst_frame = fa["frame"]
            player_d.append(d)
        if fa["puck"] and fb["puck"]:
            puck_d.append(math.dist(fa["puck"]["rink_xy"], fb["puck"]["rink_xy"]))

    def pct(x):
        return f"{len(x)}/{n} ({100 * len(x) / n:.2f}%)"

    player_d.sort()
    print(f"frames: {n}")
    print(f"fit present differs:      {pct(fit_diff)}  {fit_diff[:10]}")
    print(f"keypoints used differ:    {pct(kp_diff)}  {kp_diff[:10]}")
    print(f"view differs:             {pct(view_diff)}  {view_diff[:10]}")
    print(f"rejection reason differs: {pct(rejected_diff)}  (informational: both may reject, for different reasons)")
    print("on frames whose fits agree:")
    print(f"  frames whose projected players differ: {player_count_diff}")
    if player_d:
        print(f"  player position difference, ft: median {player_d[len(player_d) // 2]:.5f}, "
              f"p99 {player_d[int(0.99 * (len(player_d) - 1))]:.5f}, max {player_d[-1]:.5f} "
              f"(frame {worst_frame}) over {len(player_d)} positions")
    if puck_d:
        print(f"  puck position difference, ft: max {max(puck_d):.5f} over {len(puck_d)} positions")

    worst = max(len(fit_diff), len(kp_diff)) / n
    if worst > args.max_mismatch or (player_d and player_d[int(0.99 * (len(player_d) - 1))] > args.max_ft):
        sys.exit("MISMATCH")
    print("OK")


if __name__ == "__main__":
    main()
