"""Compare two detections.json files (run_models.py vs hockey-vision detect) frame by frame.

    python rust/bench/compare_detections.py <a/detections.json> <b/detections.json> [--frames N]

Reports identical frame records, then per model: boxes matched one to one (same class, IoU >= 0.5,
greedy by IoU), unmatched boxes, and how far matched boxes and confidences differ. For players
also track ids (whether matched boxes carry the same id, after mapping ids one to one), jersey
numbers and readings; and camera cuts.
"""

import argparse
import json
from collections import Counter, defaultdict


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(da, db):
    pairs = sorted(((iou(x["box_xyxy"], y["box_xyxy"]), i, j) for i, x in enumerate(da) for j, y in enumerate(db)
                    if x["class"] == y["class"]), reverse=True)
    used_a, used_b, out = set(), set(), []
    for v, i, j in pairs:
        if v < 0.5 or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j))
    return out


def load(path, n):
    frames = json.loads(open(path).read())
    return frames[:n] if n else frames


def compare(A, B):
    """Statistics comparing two loaded detections.json lists (see the module docstring):
    {"frames", "identical", "cut_diff", "models": {model: Counter}, "worst": {model: [box, conf]},
    "ids_consistent", "ids_total", "tracks_a", "tracks_b"}."""
    n = min(len(A), len(B))
    same = sum(json.dumps(A[k]) == json.dumps(B[k]) for k in range(n))
    cuts = sum(A[k].get("cut") != B[k].get("cut") for k in range(n))

    stats = defaultdict(Counter)
    worst = defaultdict(lambda: [0.0, 0.0])
    id_map, id_votes = {}, defaultdict(Counter)
    for k in range(n):
        da, db = A[k]["detections"], B[k]["detections"]
        for model in sorted(set(da) | set(db)):
            xa, xb = da.get(model, []), db.get(model, [])
            s = stats[model]
            s["a"] += len(xa)
            s["b"] += len(xb)
            m = match(xa, xb)
            s["matched"] += len(m)
            for i, j in m:
                x, y = xa[i], xb[j]
                d = max(abs(p - q) for p, q in zip(x["box_xyxy"], y["box_xyxy"]))
                worst[model][0] = max(worst[model][0], d)
                worst[model][1] = max(worst[model][1], abs(x["confidence"] - y["confidence"]))
                s["exact"] += x == y
                s["box_exact"] += d == 0
                if model == "player":
                    if x.get("track_id") is not None and y.get("track_id") is not None:
                        id_votes[x["track_id"]][y["track_id"]] += 1
                    s["number_same"] += x.get("jersey_number") == y.get("jersey_number")
                    s["number_both"] += x.get("jersey_number") is not None or y.get("jersey_number") is not None
                    s["reading_same"] += x.get("jersey_reading") == y.get("jersey_reading")
    for a, votes in id_votes.items():
        id_map[a] = votes.most_common(1)[0][0]
    # Track consistency: of matched player boxes with ids, how many carry the mapped id.
    consistent = total = 0
    for a, votes in id_votes.items():
        consistent += votes[id_map[a]]
        total += sum(votes.values())
    return {"frames": n, "identical": same, "cut_diff": cuts, "models": dict(stats), "worst": dict(worst),
            "ids_consistent": consistent, "ids_total": total, "tracks_a": len(id_votes),
            "tracks_b": len({v for c in id_votes.values() for v in c})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()
    A, B = load(args.a, args.frames), load(args.b, args.frames)
    if len(A) != len(B):
        print(f"frame counts differ: {len(A)} vs {len(B)}; comparing the first {min(len(A), len(B))}")
    r = compare(A, B)
    n, same, stats, worst = r["frames"], r["identical"], r["models"], r["worst"]
    consistent, total, id_votes_n = r["ids_consistent"], r["ids_total"], (r["tracks_a"], r["tracks_b"])
    print(f"identical frame records: {same}/{n} ({100 * same / n:.1f}%)")
    print(f"camera cuts: {sum(bool(f.get('cut')) for f in A[:n])} vs {sum(bool(f.get('cut')) for f in B[:n])}, "
          f"differ on {r['cut_diff']} frames")
    for model, s in stats.items():
        line = (f"{model:7} boxes {s['a']:6} vs {s['b']:6}; matched {s['matched']:6} "
                f"({100 * s['matched'] / max(1, max(s['a'], s['b'])):.2f}%), identical {s['exact']:6}, "
                f"max box diff {worst[model][0]:.1f} px, max conf diff {worst[model][1]:.4f}")
        print(line)
    p = stats.get("player")
    if p:
        print(f"players: track ids consistent on {consistent}/{total} matched boxes "
              f"({100 * consistent / max(1, total):.2f}%), tracks {id_votes_n[0]} vs {id_votes_n[1]}; "
              f"jersey number same on {p['number_same']}/{p['matched']} "
              f"(of {p['number_both']} with a number on either side); reading same on {p['reading_same']}/{p['matched']}")


if __name__ == "__main__":
    main()
