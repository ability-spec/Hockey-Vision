"""Import `hockey-vision export` outputs into the web app's database, one game per export folder.

    python -m webapp.ingest outputs/<clip>/export
    python -m webapp.ingest outputs/<clip>/export --roster rosters/2026-03-03_MTL_at_SJS.csv
    python -m webapp.ingest outputs/<clip>/export --date 2026-03-03 --away MTL --home SJS --game-id 2025020969

Players are matched to NHL player ids by team and jersey number (the name breaks ties), so the clip
must have been run with --teams (hockey-vision run / detect). With --roster (the fetch_roster.py --game file the
clip was run with) the game's date and teams come from it and players are matched against exactly
who dressed; otherwise against the active rosters. Importing the same folder again replaces it.
The game's NHL game id is looked up from its date and teams (or given with --game-id), and its
official box score fetched (nhl.py), so the players' scores combine the video with the results.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date as Date
from pathlib import Path

try:
    from . import db, nhl
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from webapp import db, nhl

GAME_ROSTER = re.compile(r"(\d{4}-\d{2}-\d{2})_([A-Z]{3})_at_([A-Z]{3})$")
TEXT_COLUMNS = {"player_id", "team", "team_abbrev", "jersey_number", "player_name", "role", "identified"}


def read_player_stats(path):
    """player_stats.csv rows with numbers parsed."""
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        out = {}
        for k, v in r.items():
            if k in TEXT_COLUMNS or v == "":
                out[k] = v
            else:
                try:
                    out[k] = float(v) if "." in v else int(v)
                except ValueError:
                    out[k] = v
        out["identified"] = str(r.get("identified", "")).lower() in ("true", "1")
        rows.append(out)
    return rows


def _lookup(con, roster_rows=None):
    """{(team, number): [(player_id, name)]} from a game roster, or the database's players."""
    rows = roster_rows if roster_rows is not None else [
        {"team": r["team"], "number": r["number"], "player_id": r["player_id"], "player": r["name"]}
        for r in con.execute("SELECT * FROM players")]
    out = defaultdict(list)
    for r in rows:
        out[(r["team"], int(r["number"]))].append((int(r["player_id"]), r["player"]))
    return out


def match_player(lookup, row):
    """The NHL player id for a player_stats.csv row, or None."""
    if not row.get("identified") or not row.get("team_abbrev") or row.get("jersey_number") in ("", None):
        return None
    candidates = lookup.get((row["team_abbrev"], int(row["jersey_number"])), [])
    if len(candidates) == 1:
        return candidates[0][0]
    name = (row.get("player_name") or "").strip().lower()
    named = [pid for pid, n in candidates if n.lower() == name]
    return named[0] if len(named) == 1 else None


def goals_against(rows):
    """Each team's possible goals against, credited to its goalie seen the longest."""
    by_team = defaultdict(float)
    for r in rows:
        by_team[r.get("team_abbrev")] += r.get("possible_goals") or 0
    goalies = {}
    for r in rows:
        if r.get("role") == "goalie" and r.get("team_abbrev"):
            best = goalies.get(r["team_abbrev"])
            if best is None or (r.get("time_detected_s") or 0) > (best.get("time_detected_s") or 0):
                goalies[r["team_abbrev"]] = r
    for team, g in goalies.items():
        g["goals_against"] = sum(v for t, v in by_team.items() if t != team)


def import_game(con, rows, source, date, away=None, home=None, label=None, nhl_game_id=None, is_demo=False,
                roster_rows=None):
    """Store one game's player_stats rows. Returns (game id, players matched, rows skipped)."""
    goals_against(rows)
    lookup = _lookup(con, roster_rows)
    teams = sorted({r["team_abbrev"] for r in rows if r.get("team_abbrev")})
    if not (away and home) and len(teams) == 2:
        away, home = teams
    label = label or (f"{away} at {home}" if away and home else Path(source).name)
    matched, skipped = {}, 0
    for r in rows:
        pid = match_player(lookup, r)
        if pid is None:
            skipped += 1
        elif pid not in matched or r.get("time_detected_s", 0) > matched[pid].get("time_detected_s", 0):
            matched[pid] = r
    with con:
        con.execute("DELETE FROM games WHERE source = ?", (source,))
        game_id = con.execute(
            "INSERT INTO games (source, label, date, nhl_game_id, away, home, is_demo, imported_at) VALUES (?,?,?,?,?,?,?,?)",
            (source, label, date, nhl_game_id, away, home, int(is_demo), db.now())).lastrowid
        con.executemany("INSERT INTO game_stats (game_id, player_id, team, role, stats) VALUES (?,?,?,?,?)",
                        [(game_id, pid, r.get("team_abbrev"), r.get("role") or "skater", json.dumps(r))
                         for pid, r in matched.items()])
    return game_id, len(matched), skipped


def import_export(con, export_dir, date=None, away=None, home=None, label=None, nhl_game_id=None, roster=None):
    export_dir = Path(export_dir)
    stats_csv = export_dir / "player_stats.csv"
    if not stats_csv.exists():
        sys.exit(f"No player_stats.csv in {export_dir} (run hockey-vision export first)")
    roster_rows = None
    if roster:
        roster_rows = db.load_roster(con, roster)
        m = GAME_ROSTER.search(Path(roster).stem)
        if m:
            date, away, home = date or m[1], away or m[2], home or m[3]
        date = date or (roster_rows[0]["as_of"] if roster_rows else None)
    rows = read_player_stats(stats_csv)
    if not any(r.get("team_abbrev") for r in rows):
        sys.exit(f"{stats_csv} has no team abbreviations: run hockey-vision with --teams so players can be named")
    return import_game(con, rows, str(export_dir.resolve()), date or Date.today().isoformat(), away, home, label,
                       nhl_game_id, roster_rows=roster_rows)


def main():
    parser = argparse.ArgumentParser(description="Import hockey-vision export outputs into the Hockey-Vision web app.")
    parser.add_argument("exports", type=Path, nargs="+", help="Export folders (outputs/<clip>/export)")
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DB, help="Database file (default: %(default)s)")
    parser.add_argument("--roster", type=Path, help="The game's roster CSV from fetch_roster.py --game")
    parser.add_argument("--date", help="Game date, YYYY-MM-DD (default: from --roster, else today)")
    parser.add_argument("--away", help="Away team abbreviation")
    parser.add_argument("--home", help="Home team abbreviation")
    parser.add_argument("--label", help="Name shown for the game (default: '<away> at <home>')")
    parser.add_argument("--game-id", type=int, help="NHL game id")
    args = parser.parse_args()

    con = db.connect(args.db)
    db.ensure_players(con)
    for export_dir in args.exports:
        game_id, matched, skipped = import_export(con, export_dir, args.date, args.away, args.home, args.label,
                                                  args.game_id, args.roster)
        game = con.execute("SELECT label, date FROM games WHERE id = ?", (game_id,)).fetchone()
        print(f"{export_dir}: game {game_id} ({game['label']}, {game['date']}): {matched} players"
              + (f", {skipped} rows not matched to a player (unidentified or not on the roster)" if skipped else ""))
        link_box_score(con, game_id, args.game_id)


def link_box_score(con, game_id, nhl_game_id=None):
    """Find the game's NHL game id (unless given) and fetch its box score, so its scores combine the
    video with the official results. Failing only prints: `python -m webapp.nhl sync` retries."""
    g = con.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    try:
        nhl_game_id = nhl_game_id or g["nhl_game_id"] or (
            nhl.find_game_id(g["date"], g["away"], g["home"]) if g["away"] and g["home"] else None)
        if not nhl_game_id:
            print("  No NHL game found for its date and teams: scored from the video alone")
            return
        with con:
            con.execute("UPDATE games SET nhl_game_id = ? WHERE id = ?", (nhl_game_id, game_id))
        box, lines = nhl.sync_boxscore(con, nhl_game_id)
        print(f"  NHL game {nhl_game_id}: {box['away']} {box['away_score']} at {box['home']} {box['home_score']}, "
              f"{len(lines)} player lines; scores combine the video with the box score")
    except nhl.NhlError as e:
        print(f"  Couldn't reach the NHL's API ({e}); `python -m webapp.nhl sync` retries it")


if __name__ == "__main__":
    main()
