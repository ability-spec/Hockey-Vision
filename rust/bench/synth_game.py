"""Synthetic detections.json + positions.json for exercising export_data.py / events.py: players
skating, goalies at their nets, a puck that is carried, passed, stolen and shot, camera cuts, frames
without a rink fit and unread jersey numbers. Not realistic video, but it produces thousands of
possessions and events, so it tests the event rules much harder than a 10-second clip does.

    python rust/bench/synth_game.py <out clip folder> [--frames 30000] [--seed 0]
"""

import argparse
import json
import math
import random
from pathlib import Path

FPS = 30.0
NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliett",
         "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Slafkovský", "O'Neil, Jr."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # 5 skaters + a goalie per team, plus a referee. A's goalie guards the left net in the first
    # half and the right one after (teams switch ends), so the attacking direction changes.
    players = []
    for team, cls, gcls, abbrev in (("A", "team_a_player", "goalie_a", "SJS"), ("B", "team_b_player", "goalie_b", "MTL")):
        for k in range(5):
            players.append({"team": team, "cls": cls, "abbrev": abbrev, "number": str(rnd.randint(2, 98)),
                            "x": rnd.uniform(-80, 80), "y": rnd.uniform(-35, 35), "vx": 0.0, "vy": 0.0})
        players.append({"team": team, "cls": gcls, "abbrev": abbrev, "number": str(rnd.randint(1, 40)),
                        "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "goalie": True})
    players.append({"team": "", "cls": "referee", "abbrev": None, "number": None,
                    "x": 0.0, "y": 30.0, "vx": 0.0, "vy": 0.0})
    for p in players:
        p["name"] = rnd.choice(NAMES) if p["number"] else None

    next_track = 1
    for p in players:
        p["track"] = next_track
        next_track += 1

    carrier = 0
    puck = [players[0]["x"], players[0]["y"]]
    puck_v = [0.0, 0.0]
    flight = 0  # frames the puck keeps flying
    det_frames, pos_frames = [], []
    for i in range(args.frames):
        t = round(i / FPS, 3)
        half = i >= args.frames // 2
        cut = i > 0 and rnd.random() < 1 / 900
        if cut or rnd.random() < 1 / 2000:  # new tracks: ids restart for some players (tracker handover)
            for p in rnd.sample(players, 3):
                p["track"] = next_track
                next_track += 1

        for p in players:
            if p.get("goalie"):
                side = -1 if (p["team"] == "A") != half else 1
                p["x"] = side * rnd.uniform(84, 88)
                p["y"] = rnd.uniform(-3, 3)
                continue
            target = puck if rnd.random() < 0.6 else (rnd.uniform(-90, 90), rnd.uniform(-38, 38))
            ax, ay = target[0] - p["x"], target[1] - p["y"]
            n = math.hypot(ax, ay) or 1
            p["vx"] = 0.9 * p["vx"] + 0.1 * 25 * ax / n + rnd.gauss(0, 2)
            p["vy"] = 0.9 * p["vy"] + 0.1 * 25 * ay / n + rnd.gauss(0, 2)
            p["x"] = max(-99, min(99, p["x"] + p["vx"] / FPS))
            p["y"] = max(-41, min(41, p["y"] + p["vy"] / FPS))

        if flight > 0:
            puck[0] += puck_v[0] / FPS
            puck[1] += puck_v[1] / FPS
            flight -= 1
            if flight == 0:  # whoever is nearest picks it up
                carrier = min(range(len(players) - 1), key=lambda k: math.hypot(players[k]["x"] - puck[0],
                                                                                   players[k]["y"] - puck[1]))
        else:
            c = players[carrier]
            puck = [c["x"] + rnd.uniform(-2, 2), c["y"] + rnd.uniform(-2, 2)]
            r = rnd.random()
            if r < 0.02:  # pass or shot
                if rnd.random() < 0.35 and abs(c["x"]) < 80:
                    goal = 89 * rnd.choice((-1, 1))
                    gy = rnd.uniform(-5, 5)
                    speed = rnd.uniform(50, 140)
                else:
                    mate = rnd.choice([k for k, p in enumerate(players[:-1]) if k != carrier])
                    goal, gy = players[mate]["x"], players[mate]["y"]
                    speed = rnd.uniform(30, 90)
                dx, dy = goal - puck[0], gy - puck[1]
                n = math.hypot(dx, dy) or 1
                puck_v = [speed * dx / n, speed * dy / n]
                flight = max(2, int(n / speed * FPS))
            elif r < 0.03:  # stolen by the nearest opponent
                opp = [k for k, p in enumerate(players[:-1]) if p["team"] != c["team"]]
                carrier = min(opp, key=lambda k: math.hypot(players[k]["x"] - c["x"], players[k]["y"] - c["y"]))
                players[carrier]["x"], players[carrier]["y"] = c["x"] + rnd.uniform(-5, 5), c["y"] + rnd.uniform(-5, 5)

        fit = rnd.random() > 0.08
        dets, pos_players = [], []
        for k, p in enumerate(players):
            if rnd.random() < 0.05:
                continue  # missed detection
            # A fake image box: anything unique per player works, export_data only matches boxes.
            x1 = round(960 + p["x"] * 9 + rnd.uniform(-0.04, 0.04), 1)
            y1 = round(540 - p["y"] * 9, 1)
            box = [x1, y1, round(x1 + 40.0 + k, 1), round(y1 + 90.0, 1)]
            known = p["number"] and (i // 300 + k) % 4 != 0  # numbers sometimes not read yet
            d = {"class": p["cls"], "confidence": round(rnd.uniform(0.3, 0.95), 4), "box_xyxy": box,
                 "track_id": p["track"], "jersey_number": p["number"] if known else None,
                 "team": p["abbrev"], "player_name": p["name"] if known else None, "jersey_reading": None}
            if rnd.random() < 0.01:
                del d["track_id"]
            dets.append(d)
            if fit:
                jx, jy = rnd.gauss(0, 0.3), rnd.gauss(0, 0.3)
                pos_players.append({"class": p["cls"], "confidence": d["confidence"], "jersey_number": d["jersey_number"],
                                    "player_name": d["player_name"], "track_id": d.get("track_id"), "box_xyxy": box,
                                    "rink_xy": [round(p["x"] + jx, 2), round(p["y"] + jy, 2)]})
        pucks = []
        puck_pos = None
        if rnd.random() < 0.55:
            px1 = round(960 + puck[0] * 9, 1)
            pucks.append({"class": "puck", "confidence": round(rnd.uniform(0.6, 0.99), 4),
                          "box_xyxy": [px1, 500.0, round(px1 + 8, 1), 508.0]})
            if fit and abs(puck[0]) <= 100 and abs(puck[1]) <= 42.5:
                puck_pos = {"confidence": pucks[0]["confidence"],
                            "rink_xy": [round(puck[0] + rnd.gauss(0, 0.2), 2), round(puck[1] + rnd.gauss(0, 0.2), 2)]}
        det_frames.append({"frame": i, "time_s": t, "cut": cut, "detections": {"player": dets, "puck": pucks}})
        view = "side" if rnd.random() > 0.03 else "end"
        pos_frames.append({"frame": i, "time_s": t, "homography": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]] if fit else None,
                           "cut": cut, "view": view, "rejected": None if fit or rnd.random() < 0.5 else "player_scale",
                           "keypoints_used": ["a", "b", "c", "d"] if fit else [],
                           "players": pos_players, "puck": puck_pos})

    with open(args.out / "detections.json", "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(r) for r in det_frames) + "\n]\n")
    (args.out / "positions.json").write_text(json.dumps(pos_frames))
    print(f"Wrote {args.frames} frames to {args.out}")


if __name__ == "__main__":
    main()
