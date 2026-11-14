"""Fetch NHL rosters (jersey numbers and names) from the NHL's public API (api-web.nhle.com) for
hockey-vision's --teams / --roster (run / detect), which limit jersey readings to the two teams' numbers and
name the players.

    python fetch_roster.py                          # every team's active roster
                                                    #   -> rosters/nhl_active_players.csv
    python fetch_roster.py --game 2025020969        # the players who dressed for one game
    python fetch_roster.py --game 2026-03-03 MTL SJS    (or found by date and teams)
                                                    #   -> rosters/<date>_<away>_at_<home>.csv

Both files have one row per player with the same columns: team (abbreviation, e.g. SJS), team_name,
number, player, position, player_id, as_of. Active rosters change through the season (and are
larger in training camp), so refetch before processing new games; for a past game use --game,
which lists exactly who played.
"""

import argparse
import csv
import datetime
import json
import sys
import urllib.request
from pathlib import Path

API = "https://api-web.nhle.com/v1"
ROSTERS_DIR = Path(__file__).parent / "rosters"
ACTIVE_ROSTER = ROSTERS_DIR / "nhl_active_players.csv"
COLUMNS = ["team", "team_name", "number", "player", "position", "player_id", "as_of"]
POSITIONS = {"C": "Center", "L": "Left wing", "R": "Right wing", "D": "Defense", "G": "Goalie"}


def get(path):
    # The API turns away Python's default User-Agent.
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "Mozilla/5.0 (Hockey-Vision roster fetch)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def teams():
    """{abbreviation: full name} for every current team."""
    return {t["teamAbbrev"]["default"]: t["teamName"]["default"] for t in get("standings/now")["standings"]}


def active_rosters():
    today = datetime.date.today().isoformat()
    rows = []
    for abbrev, name in sorted(teams().items()):
        roster = get(f"roster/{abbrev}/current")
        for group in roster.values():
            for p in group:
                if "sweaterNumber" not in p:
                    continue
                rows.append({"team": abbrev, "team_name": name, "number": p["sweaterNumber"],
                             "player": f"{p['firstName']['default']} {p['lastName']['default']}",
                             "position": POSITIONS.get(p.get("positionCode"), p.get("positionCode", "")),
                             "player_id": p["id"], "as_of": today})
        print(f"  {abbrev}: {sum(r['team'] == abbrev for r in rows)} players")
    return rows, ACTIVE_ROSTER


def find_game(date, team1, team2):
    for day in get(f"schedule/{date}")["gameWeek"]:
        if day["date"] != date:
            continue
        for g in day["games"]:
            if {g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]} == {team1, team2}:
                return g["id"]
    sys.exit(f"No {team1} vs {team2} game on {date}")


def player_name(p):
    """A box score player's full name from their player page, or the box score's 'F. Last'."""
    try:
        info = get(f"player/{p['playerId']}/landing")
        return f"{info['firstName']['default']} {info['lastName']['default']}"
    except (OSError, KeyError):
        return p["name"]["default"]


def game_roster(game_id):
    box = get(f"gamecenter/{game_id}/boxscore")
    rows = []
    for side in ("awayTeam", "homeTeam"):
        team = box[side]
        name = f"{team['placeName']['default']} {team['commonName']['default']}"
        for group in ("forwards", "defense", "goalies"):
            for p in box["playerByGameStats"][side].get(group, []):
                pos = p.get("position", "G" if group == "goalies" else "")
                rows.append({"team": team["abbrev"], "team_name": name, "number": p["sweaterNumber"],
                             "player": player_name(p), "position": POSITIONS.get(pos, pos),
                             "player_id": p["playerId"], "as_of": box["gameDate"]})
    away, home = box["awayTeam"]["abbrev"], box["homeTeam"]["abbrev"]
    print(f"Game {game_id}: {away} {box['awayTeam'].get('score', '')} at {home} "
          f"{box['homeTeam'].get('score', '')}, {box['gameDate']}")
    return rows, ROSTERS_DIR / f"{box['gameDate']}_{away}_at_{home}.csv"


def main():
    parser = argparse.ArgumentParser(description="Fetch NHL rosters with jersey numbers.")
    parser.add_argument("--game", nargs="+", metavar="GAME",
                        help="One game's dressed players: an NHL game ID, or a date (YYYY-MM-DD) and the two "
                             "team abbreviations (default: every team's active roster)")
    args = parser.parse_args()

    if args.game:
        game_id = args.game[0] if args.game[0].isdigit() else find_game(*args.game[:3])
        rows, out = game_roster(game_id)
    else:
        rows, out = active_rosters()
    rows.sort(key=lambda r: (r["team"], int(r["number"])))
    ROSTERS_DIR.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} players, {len({r['team'] for r in rows})} teams -> {out}")


if __name__ == "__main__":
    main()
