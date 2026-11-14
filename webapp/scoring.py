"""Hockey-Vision scores: how well a player played in one game (0-100), their form over recent games,
whether a pool should keep or trade them, and how strong a pool is overall.

Everything here is a pure function of a player's player_stats.csv row (hockey-vision export, the video)
and their official box score line (nhl.py), so the rules can be read and tuned in one place.

Game score
    Built from up to three parts, depending on what the game has:

    play      from the video: movement, possession, pass quality and pressure (PLAY_WEIGHTS: a
              takeaway +2, a good pass +1, a turnover -2, ...). Broadcast video only shows each
              player some of the time, so the points become a rate per minute detected, shrunk
              towards the league's typical rate (SHRINK_MINUTES) so 20 seconds on screen can't
              decide a game.
    results   from the NHL's box score: goals, assists, shots on goal, +/-, hits, blocks, penalty
              minutes (RESULT_WEIGHTS); goalies saves, goals against, the win, shutouts.
    backed    plays the video shows behind an official result, in the same game (SYNERGY): assists
              in a game the video saw good passes, goals it saw the shots for, points after it saw
              the player win the puck, a plus backed by defensive plays; and turnovers in a minus
              game count against.

    Each part is compared with every other player-game in the database of the same role: the
    video rates by median and spread, the results by where the game ranks among them (a
    percentile, since a few multi-point nights would swamp a spread). Until the database has
    enough games (MIN_CALIBRATION_ROWS, MIN_RESULT_ROWS) it uses DEFAULT_CALIBRATION, the results
    part measured on 100 of the 2026-27 season's NHL box scores. The two are blended (W_PLAY,
    W_RESULT; the play part counts less when the player was barely on screen). With no box score the score is the video alone, with
    shots and possible goals in place of the official results (VIDEO_WEIGHTS); with no video row
    (a player whose number was never read) it is the results alone. The blend is mapped onto 0-100
    with a logistic curve: 50 is a typical game, 73 one spread better, 88 two.

Form
    A recency-weighted mean of the last FORM_GAMES scores, games with a box score or more time
    detected counting more. Trend is the last 3 games against the ones before them.

Trade call (per pool player)
    hold    fewer than MIN_GAMES games, too early to judge
    trade   form below TRADE_BELOW
    watch   form below WATCH_BELOW, or a trend drop of TREND_DROP or more
    keep    otherwise ("core" from CORE_ABOVE)

Pool assessment
    Strength (mean form of the pool's rated players, graded A-F), depth per position group against
    the pool's slots, trade candidates and the best available upgrades, all folded into a 0-100
    trade urgency and a verdict: hold, make a targeted trade, or be active.
"""

import math
from bisect import bisect_left, bisect_right
from statistics import NormalDist, mean, median, quantiles

# From the video: how the player played. Points per counted stat, as a rate per minute detected.
PLAY_WEIGHTS = {
    "skater": {
        "good_passes": ("Good passes", 1.0),
        "passes_completed": ("Passes completed", 0.5),
        "passes_received": ("Passes received", 0.2),
        "possession_time_s": ("Possession time (s)", 0.05),
        "distance_ft": ("Distance skated (ft)", 0.0005),
        "takeaways": ("Takeaways", 2.0),
        "interceptions": ("Interceptions", 1.5),
        "defensive_plays": ("Defensive plays", 1.0),
        "turnovers": ("Turnovers", -2.0),
        "times_defended": ("Times defended", -0.3),
    },
    "goalie": {
        "goalie_recoveries": ("Puck recoveries", 0.3),
        "passes_completed": ("Passes completed", 0.2),
        "turnovers": ("Turnovers", -1.0),
    },
}
# The video's own shot and save counts, used in place of the box score when a game has none. Shots
# on net and possible goals are also counted in shots, so a possible goal is worth 0.5 + 1.5 + 5.
VIDEO_RESULT_WEIGHTS = {
    "skater": {
        "possible_goals": ("Possible goals", 5.0),
        "shots_on_net": ("Shots on net", 1.5),
        "shots": ("Shots", 0.5),
        "blocked_shots": ("Blocked shots", 1.5),
        "shots_blocked_by_opponent": ("Shots blocked by opponent", -0.5),
    },
    "goalie": {
        "saves": ("Saves", 1.0),
        "goals_against": ("Possible goals against", -3.0),
    },
}
VIDEO_WEIGHTS = {role: {**PLAY_WEIGHTS[role], **VIDEO_RESULT_WEIGHTS[role]} for role in PLAY_WEIGHTS}
# From the box score: what the player produced, per game.
RESULT_WEIGHTS = {
    "skater": {
        "goals": ("Goals", 6.0),
        "assists": ("Assists", 4.0),
        "pp_goals": ("Power-play goals (extra)", 1.0),
        "shots": ("Shots on goal", 0.5),
        "plus_minus": ("Plus/minus", 1.0),
        "hits": ("Hits", 0.3),
        "blocks": ("Blocked shots", 0.5),
        "pim": ("Penalty minutes", -0.25),
    },
    "goalie": {
        "saves": ("Saves", 0.25),
        "goals_against": ("Goals against", -2.0),
    },
}
GOALIE_DECISION = {"W": ("Win", 3.0), "O": ("Overtime loss", 1.0)}
SHUTOUT = ("Shutout", 3.0)
SHUTOUT_MIN_TOI_S = 30 * 60
# Video x box score, per game: (label, official stat, video stats, points each). The count is the
# smaller of the two sides, so each official result needs its own play on video to count.
SYNERGY = {
    "skater": [
        ("Assists backed by good passes on video", "assists", ("good_passes",), 3.0),
        ("Goals backed by shots on video", "goals", ("shots",), 3.0),
        ("Points after winning the puck on video", "points", ("takeaways", "interceptions"), 2.0),
        ("Plus backed by defensive plays on video", "plus", ("defensive_plays",), 1.5),
        ("Turnovers on video in a minus game", "minus", ("turnovers",), -2.0),
    ],
    "goalie": [
        ("Rebound control in a game of 2 or fewer against", "low_goals_against", ("goalie_recoveries",), 0.5),
    ],
}
SYNERGY_MAX_EACH = 6

W_PLAY, W_RESULT = 0.45, 0.55

# The video parts: typical points per minute detected and their spread. The results part: the
# 0th, 5th, ..., 100th percentiles of results points per game, measured on 100 of the 2026-27
# season's box scores (3,600 skater games, 199 goalie games of 30+ minutes; no video, so nothing
# backed). The database's own games replace these once it has MIN_CALIBRATION_ROWS (video) or
# MIN_RESULT_ROWS (results) of a role.
DEFAULT_CALIBRATION = {
    "video": {"skater": (1.5, 1.2), "goalie": (0.8, 0.8)},
    "play": {"skater": (1.0, 0.9), "goalie": (0.15, 0.15)},
    "results": {
        "skater": [-4.75, -1.0, -0.5, 0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.3, 2.8, 3.5, 4.8, 5.92, 7.0,
                   8.1, 9.5, 13.0, 29.9],
        "goalie": [-9.25, -6.53, -4.8, -3.25, -2.0, -1.5, -0.65, 0.33, 1.5, 2.0, 2.75, 3.48, 4.0, 4.5, 5.0,
                   5.25, 5.75, 6.25, 7.25, 8.28, 13.5],
    },
}
MIN_CALIBRATION_ROWS = 30
MIN_RESULT_ROWS = 200
MIN_CALIBRATION_MINUTES = 0.5
MIN_SPREAD = {"video": {"skater": 0.3, "goalie": 0.2}, "play": {"skater": 0.3, "goalie": 0.1}}
PERCENTILE_LIMIT = 0.005   # results percentiles are kept within 0.5-99.5% (z within +-2.58)
SHRINK_MINUTES = 3.0       # minutes of league-typical play every video rate is blended with
FULL_MINUTES = 4.0         # minutes detected for the video to count fully and be "high" confidence

FORM_GAMES = 5
FORM_DECAY = 0.8           # each older game counts this much of the one after it
MIN_GAMES = 3
TRADE_BELOW = 40
WATCH_BELOW = 46
CORE_ABOVE = 60
TREND_DROP = -10

UPGRADE_MARGIN = 8         # form points an available player must beat a pool player by
GRADES = [(58, "A"), (54, "B"), (50, "C"), (46, "D"), (-1, "F")]
POSITION_GROUPS = {"Center": "F", "Left wing": "F", "Right wing": "F", "Defense": "D", "Goalie": "G"}
GROUP_NAMES = {"F": "Forwards", "D": "Defense", "G": "Goalies"}
GROUP_UNITS = {"F": ("forward", "forwards"), "D": ("defenseman", "defensemen"), "G": ("goalie", "goalies")}
PARTS = {"play": "From the video: how they played", "results": "From the NHL: what they produced",
         "backed": "Backed by video: results the video saw them create", "video": "From the video"}


def position_group(position):
    return POSITION_GROUPS.get(position, "F")


def _role(role):
    return "goalie" if role == "goalie" else "skater"


def _count(stats, weights, part):
    components, total = [], 0.0
    for key, (label, weight) in weights.items():
        count = float(stats.get(key) or 0)
        if count:
            components.append({"part": part, "stat": key, "label": label, "count": round(count, 2),
                               "weight": weight, "points": round(count * weight, 2)})
            total += count * weight
    return total, components


def video_points(stats, role):
    """(points, components) of the video alone, for a game with no box score."""
    return _count(stats, VIDEO_WEIGHTS[_role(role)], "video")


def play_points(stats, role):
    """(points, components) of the video's play stats, for a game with a box score."""
    return _count(stats, PLAY_WEIGHTS[_role(role)], "play")


def result_points(line):
    """(points, components) of an official box score line."""
    role = _role(line.get("role"))
    total, components = _count(line, RESULT_WEIGHTS[role], "results")
    if role == "goalie":
        extra = []
        if line.get("decision") in GOALIE_DECISION:
            extra.append(GOALIE_DECISION[line["decision"]])
        if line.get("goals_against") == 0 and (line.get("toi_s") or 0) >= SHUTOUT_MIN_TOI_S:
            extra.append(SHUTOUT)
        for label, pts in extra:
            components.append({"part": "results", "stat": label.lower(), "label": label, "count": 1,
                               "weight": pts, "points": pts})
            total += pts
    return total, components


def synergy_points(stats, line):
    """(points, components) for official results the video shows the plays behind, in one game."""
    role = _role(line.get("role"))
    pm = line.get("plus_minus") or 0
    official = {"assists": line.get("assists") or 0, "goals": line.get("goals") or 0,
                "points": line.get("points") or 0, "plus": max(pm, 0), "minus": max(-pm, 0),
                "low_goals_against": SYNERGY_MAX_EACH if (line.get("goals_against") or 0) <= 2
                and (line.get("toi_s") or 0) >= SHUTOUT_MIN_TOI_S else 0}
    components, total = [], 0.0
    for label, result, video, weight in SYNERGY[role]:
        count = min(official[result], sum(float(stats.get(k) or 0) for k in video), SYNERGY_MAX_EACH)
        if count > 0:
            components.append({"part": "backed", "stat": result, "label": label, "count": round(count, 2),
                               "weight": weight, "points": round(count * weight, 2)})
            total += count * weight
    return total, components


def minutes_detected(stats):
    return float((stats or {}).get("time_detected_s") or 0) / 60


def calibration_inputs(stats, role, line):
    """What a player-game adds to calibrate(): {kind: (points, minutes or None)}."""
    role = _role(role)
    out = {}
    if stats:
        minutes = minutes_detected(stats)
        out["play" if line else "video"] = ((play_points if line else video_points)(stats, role)[0], minutes)
    if line:
        out["results"] = (result_points(line)[0] + (synergy_points(stats, line)[0] if stats else 0), None)
    return out


def calibrate(rows):
    """{kind: {role: (typical, spread)}} from (role, calibration_inputs()) of every player-game: the
    median (per minute for the video parts) and its scaled median absolute deviation."""
    out = {kind: dict(roles) for kind, roles in DEFAULT_CALIBRATION.items()}
    for kind in out:
        for role in out[kind]:
            if kind == "results":
                values = sorted(x[kind][0] for r, x in rows if r == role and kind in x)
                if len(values) >= MIN_RESULT_ROWS:
                    out[kind][role] = [values[0]] + quantiles(values, n=20, method="inclusive") + [values[-1]]
                continue
            values = [x[kind][0] / x[kind][1] for r, x in rows
                      if r == role and kind in x and x[kind][1] >= MIN_CALIBRATION_MINUTES]
            if len(values) >= MIN_CALIBRATION_ROWS:
                mid = median(values)
                spread = 1.4826 * median(abs(v - mid) for v in values)
                out[kind][role] = (mid, max(spread, MIN_SPREAD[kind][role]))
    return out


def percentile(x, table):
    """Where `x` falls in a table of evenly spaced percentiles (0th to 100th), 0-1. A value several
    percentiles share sits in the middle of them."""
    n = len(table) - 1
    lo, hi = bisect_left(table, x), bisect_right(table, x) - 1
    if lo <= hi:
        return (lo + hi) / 2 / n
    if lo == 0:
        return 0.0
    if lo > n:
        return 1.0
    return (lo - 1 + (x - table[lo - 1]) / (table[lo] - table[lo - 1])) / n


def results_z(points, table):
    p = min(1 - PERCENTILE_LIMIT, max(PERCENTILE_LIMIT, percentile(points, table)))
    return p, NormalDist().inv_cdf(p)


def confidence(minutes, has_results=False):
    if has_results:
        return "high"
    return "high" if minutes >= FULL_MINUTES else "medium" if minutes >= 1 else "low"


def _video_z(points, minutes, typical, spread):
    rate = (points + typical * SHRINK_MINUTES) / (minutes + SHRINK_MINUTES)
    return rate, (rate - typical) / spread


def game_score(stats, role, calibration, line=None):
    """One player's 0-100 score for one game: the video row `stats` (or None), the box score `line`
    (or None), and the parts and components that make it up."""
    role = _role(role)
    minutes = minutes_detected(stats)
    parts, components, weights_z = {}, [], []
    if stats and not line:
        typical, spread = calibration["video"][role]
        points, comps = video_points(stats, role)
        rate, z = _video_z(points, minutes, typical, spread)
        parts["video"] = {"points": round(points, 2), "rate": round(rate, 3), "z": round(z, 2)}
        components += comps
        weights_z.append((1.0, z))
    if stats and line:
        typical, spread = calibration["play"][role]
        points, comps = play_points(stats, role)
        rate, z = _video_z(points, minutes, typical, spread)
        weight = W_PLAY * min(1.0, minutes / FULL_MINUTES)
        parts["play"] = {"points": round(points, 2), "rate": round(rate, 3), "z": round(z, 2), "weight": round(weight, 2)}
        components += comps
        weights_z.append((weight, z))
    if line:
        points, comps = result_points(line)
        backed, backed_comps = synergy_points(stats, line) if stats else (0.0, [])
        pct, z = results_z(points + backed, calibration["results"][role])
        parts["results"] = {"points": round(points, 2), "percentile": round(100 * pct, 1), "z": round(z, 2),
                            "weight": W_RESULT}
        parts["backed"] = {"points": round(backed, 2)}
        components += comps + backed_comps
        weights_z.append((W_RESULT, z))
    total_w = math.sqrt(sum(w * w for w, _ in weights_z))
    z = sum(w * z for w, z in weights_z) / total_w if total_w else 0.0
    basis = "video+nhl" if stats and line else "nhl" if line else "video"
    return {"score": round(100 / (1 + math.exp(-z)), 1), "basis": basis, "minutes": round(minutes, 2),
            "confidence": confidence(minutes, bool(line)), "weight": 1.0 if line else 0.5 + 0.5 * min(1.0, minutes / FULL_MINUTES),
            "points": round(sum(c["points"] for c in components), 2), "z": round(z, 2), "parts": parts,
            "components": sorted(components, key=lambda c: -abs(c["points"]))}


def form(games):
    """A player's form from their scored games, newest first: {form, season, trend, games, last}."""
    if not games:
        return {"form": None, "season": None, "trend": None, "games": 0, "last": None}
    recent = games[:FORM_GAMES]
    weights = [FORM_DECAY ** i * g.get("weight", 0.5 + 0.5 * min(1.0, g["minutes"] / FULL_MINUTES))
               for i, g in enumerate(recent)]
    value = sum(w * g["score"] for w, g in zip(weights, recent)) / sum(weights)
    trend = None
    if len(games) >= 4:
        trend = round(mean(g["score"] for g in games[:3]) - mean(g["score"] for g in games[3:3 + FORM_GAMES]), 1)
    return {"form": round(value, 1), "season": round(mean(g["score"] for g in games), 1), "trend": trend,
            "games": len(games), "last": games[0]["score"]}


def recommendation(f):
    """Keep, watch, trade or hold for a player with form summary `f`."""
    n, value, trend = f["games"], f["form"], f["trend"]
    if n == 0:
        return {"action": "hold", "label": "Hold", "reason": "No games analysed yet."}
    if n < MIN_GAMES:
        return {"action": "hold", "label": "Hold",
                "reason": f"Only {n} game{'s' * (n != 1)} analysed (form {value:.0f}); too early to judge."}
    if value < TRADE_BELOW:
        return {"action": "trade", "label": "Trade",
                "reason": f"Form {value:.0f} is well below a typical game (50) over the last {min(n, FORM_GAMES)} games."}
    if value < WATCH_BELOW:
        return {"action": "watch", "label": "Watch",
                "reason": f"Form {value:.0f} is below a typical game (50); shop this player if it doesn't pick up."}
    if trend is not None and trend <= TREND_DROP:
        return {"action": "watch", "label": "Watch",
                "reason": f"Form {value:.0f} is fine, but the last 3 games are {-trend:.0f} points below the ones before."}
    if value >= CORE_ABOVE:
        return {"action": "keep", "label": "Keep (core)", "reason": f"Form {value:.0f}: one of the pool's drivers."}
    return {"action": "keep", "label": "Keep", "reason": f"Form {value:.0f} is at or above a typical game."}


def grade(strength):
    return next(g for floor, g in GRADES if strength >= floor)


def assess_pool(pool, others, slots):
    """Assess a pool.

    `pool` and `others` are player dicts with "player_id", "name", "team", "position", "form" (a
    form() summary), "recommendation" and optionally "on_roster" (0 when on no NHL roster); `others`
    are the players outside the pool that could be traded for. `slots` is
    {"F": n, "D": n, "G": n}, how many of each the pool's league rosters.
    """
    rated = [p for p in pool if p["form"]["games"]]
    strength = round(mean(p["form"]["form"] for p in rated), 1) if rated else None
    league = sorted(p["form"]["form"] for p in pool + others if p["form"]["games"])
    league_median = median(league) if league else None
    above = sum(p["form"]["form"] >= league_median for p in rated) if league else 0

    depth = []
    for g in ("F", "D", "G"):
        have = sum(position_group(p["position"]) == g for p in pool)
        depth.append({"group": g, "name": GROUP_NAMES[g], "have": have, "need": slots.get(g, 0),
                      "gap": max(0, slots.get(g, 0) - have)})

    by_action = {a: [p for p in pool if p["recommendation"]["action"] == a] for a in ("keep", "watch", "trade", "hold")}
    # Weakest pool players first; each available player is suggested once, so the list spreads
    # over different targets rather than naming the league's best three for everyone.
    upgrades, suggested = [], set()
    for p in sorted(by_action["trade"] + by_action["watch"], key=lambda p: p["form"]["form"]):
        group = position_group(p["position"])
        targets = sorted((o for o in others if position_group(o["position"]) == group
                          and o["player_id"] not in suggested and o["form"]["games"] >= MIN_GAMES
                          and o["form"]["form"] >= p["form"]["form"] + UPGRADE_MARGIN),
                         key=lambda o: -o["form"]["form"])[:3]
        suggested.update(o["player_id"] for o in targets)
        if targets:
            upgrades.append({"out": _brief(p), "targets": [{**_brief(o), "gain": round(o["form"]["form"] - p["form"]["form"], 1)}
                                                           for o in targets]})

    reasons, urgency = [], 0.0
    if not pool:
        return {"strength": None, "grade": None, "rated": 0, "size": 0, "depth": depth, "counts": {},
                "upgrades": [], "urgency": 0, "verdict": "empty", "verdict_label": "Add players to assess this pool",
                "reasons": ["The pool has no players yet."], "league_median": league_median, "above_median": 0}
    if rated:
        n = len(rated)
        trade_share, watch_share = len(by_action["trade"]) / n, len(by_action["watch"]) / n
        urgency += 70 * trade_share + 25 * watch_share
        urgency += max(0.0, 52 - strength) * 3
        if by_action["trade"]:
            reasons.append(f"{len(by_action['trade'])} player{'s' * (len(by_action['trade']) != 1)} "
                           f"rated Trade: {', '.join(p['name'] for p in by_action['trade'][:4])}.")
        if by_action["watch"]:
            reasons.append(f"{len(by_action['watch'])} on Watch.")
        reasons.append(f"Pool strength {strength:.0f} (grade {grade(strength)}); {above} of {n} rated players "
                       f"are above the league's median form ({league_median:.0f}).")
    if upgrades:
        best = max(t["gain"] for u in upgrades for t in u["targets"])
        urgency += min(20.0, best)
        reasons.append(f"Upgrades are available: up to +{best:.0f} form at the same position.")
    gaps = [d for d in depth if d["gap"]]
    if gaps:
        urgency += 6 * sum(d["gap"] for d in gaps)
        reasons.append("Roster gaps: " + ", ".join(f"{d['gap']} {GROUP_UNITS[d['group']][d['gap'] != 1]}" for d in gaps)
                       + " short.")
    off_roster = [p for p in pool if p.get("on_roster") == 0]
    if off_roster:
        urgency += 8 * len(off_roster)
        reasons.append(f"Not on an NHL roster (sent down, waived or released): {', '.join(p['name'] for p in off_roster[:4])}.")
    unrated = len(pool) - len(rated)
    if unrated:
        reasons.append(f"{unrated} player{'s' * (unrated != 1)} with no analysed games yet.")
    urgency = round(min(100.0, urgency))

    if not rated:
        verdict, label = "hold", "Hold: no games analysed for this pool yet"
    elif urgency >= 55:
        verdict, label = "active", "Be active: trade to improve this pool"
    elif urgency >= 30 or (by_action["trade"] and upgrades):
        verdict, label = "targeted", "Make a targeted trade"
    else:
        verdict, label = "hold", "Hold: the pool is in good shape"
    return {"strength": strength, "grade": grade(strength) if strength is not None else None, "rated": len(rated),
            "size": len(pool), "league_median": league_median, "above_median": above, "depth": depth,
            "counts": {a: len(v) for a, v in by_action.items()}, "upgrades": upgrades, "urgency": urgency,
            "verdict": verdict, "verdict_label": label, "reasons": reasons}


def _brief(p):
    return {"player_id": p["player_id"], "name": p["name"], "team": p["team"], "position": p["position"],
            "form": p["form"]["form"], "games": p["form"]["games"]}
