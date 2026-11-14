"""The web app's SQLite database: every NHL player (from the roster CSVs), the games imported from
`hockey-vision export` outputs with each player's stats, favourites and playlists.

    players          one row per NHL player, keyed by the NHL's player_id
    games            one row per imported game (an export folder)
    game_stats       one row per player per game: role, team and the player_stats.csv row as JSON
    favorites        starred players
    playlists        named lists of players; a pool is a playlist with is_pool set and roster slots
    playlist_players the players in each playlist

From the NHL's API (nhl.py), refreshed by `python -m webapp.nhl sync`:

    players          also each player's headshot and bio, from the current rosters, and on_roster:
                     1 on a current NHL roster, 0 not (sent down, waived), NULL not checked yet
    nhl_standings    one row per team: record, points, streak, division rank
    nhl_season_stats one row per player: this season's official totals
    nhl_schedule     one row per scheduled NHL game
    nhl_boxscores    official box scores of imported games, and nhl_boxscore_lines one row per player
    nhl_player_landing  each player's NHL page (last 5 games, draft, career), cached when opened
    nhl_sync         when each kind of NHL data was last refreshed, and whether it worked
"""

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(__file__).resolve().parent / "hockey_vision.db"
ACTIVE_ROSTER = ROOT / "rosters" / "nhl_active_players.csv"

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    team TEXT,
    team_name TEXT,
    number INTEGER,
    position TEXT,
    as_of TEXT
);
CREATE INDEX IF NOT EXISTS players_team ON players(team);
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT UNIQUE NOT NULL,
    label TEXT NOT NULL,
    date TEXT NOT NULL,
    nhl_game_id INTEGER,
    away TEXT,
    home TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS game_stats (
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    team TEXT,
    role TEXT NOT NULL,
    stats TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS game_stats_player ON game_stats(player_id);
CREATE TABLE IF NOT EXISTS favorites (
    player_id INTEGER PRIMARY KEY REFERENCES players(player_id),
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    is_pool INTEGER NOT NULL DEFAULT 0,
    slots_f INTEGER NOT NULL DEFAULT 12,
    slots_d INTEGER NOT NULL DEFAULT 6,
    slots_g INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nhl_standings (
    team TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nhl_season_stats (
    player_id INTEGER PRIMARY KEY,
    season TEXT NOT NULL,
    role TEXT NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nhl_schedule (
    nhl_game_id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    away TEXT,
    home TEXT,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS nhl_schedule_date ON nhl_schedule(date);
CREATE TABLE IF NOT EXISTS nhl_boxscores (
    nhl_game_id INTEGER PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nhl_boxscore_lines (
    nhl_game_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    team TEXT,
    data TEXT NOT NULL,
    PRIMARY KEY (nhl_game_id, player_id)
);
CREATE TABLE IF NOT EXISTS nhl_player_landing (
    player_id INTEGER PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nhl_sync (
    kind TEXT PRIMARY KEY,
    at TEXT NOT NULL,
    ok INTEGER NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS playlist_players (
    playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    added_at TEXT NOT NULL,
    PRIMARY KEY (playlist_id, player_id)
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Player bio columns from the NHL's rosters, added to databases made before them.
PLAYER_BIO = {"headshot": "TEXT", "shoots": "TEXT", "height_in": "INTEGER", "weight_lb": "INTEGER",
              "birth_date": "TEXT", "birth_place": "TEXT"}
PLAYER_COLUMNS = {**PLAYER_BIO, "on_roster": "INTEGER"}


def connect(path=DEFAULT_DB):
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    have = {r["name"] for r in con.execute("PRAGMA table_info(players)")}
    for column, kind in PLAYER_COLUMNS.items():
        if column not in have:
            con.execute(f"ALTER TABLE players ADD COLUMN {column} {kind}")
    return con


def load_roster(con, path=ACTIVE_ROSTER):
    """Add or update every player in a fetch_roster.py CSV. Rows dated earlier than what the
    database already has (e.g. an old game's roster) don't move a player off their current team."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with con:
        con.executemany(
            """INSERT INTO players (player_id, name, team, team_name, number, position, as_of)
               VALUES (:player_id, :player, :team, :team_name, :number, :position, :as_of)
               ON CONFLICT(player_id) DO UPDATE SET name = excluded.name, team = excluded.team,
                   team_name = excluded.team_name, number = excluded.number, position = excluded.position,
                   as_of = excluded.as_of
               WHERE excluded.as_of >= COALESCE(players.as_of, '')""",
            [{**r, "player_id": int(r["player_id"]), "number": int(r["number"])} for r in rows])
    return rows


def ensure_players(con):
    """Load the active roster into an empty database."""
    if not con.execute("SELECT 1 FROM players LIMIT 1").fetchone() and ACTIVE_ROSTER.exists():
        load_roster(con)


def data_version(con):
    """Changes whenever games or box scores are imported, linked or removed, for caching scores."""
    return tuple(con.execute("""SELECT (SELECT COUNT(*) FROM games), (SELECT MAX(id) FROM games),
                                       (SELECT MAX(imported_at) FROM games),
                                       (SELECT GROUP_CONCAT(COALESCE(nhl_game_id, ''), ',') FROM games),
                                       (SELECT COUNT(*) FROM nhl_boxscores), (SELECT MAX(updated_at) FROM nhl_boxscores)""").fetchone())


def all_player_games(con):
    """Every player-game to score, newest first: each video row (stats) with its official box score
    line if the game has one (line), and the box score lines of players the video didn't identify
    (stats None). Lines of players not in the database, or with no ice time (a dressed backup
    goalie who didn't go in), are left out."""
    games = {r["id"]: dict(r) for r in con.execute("SELECT * FROM games")}
    lines = {}
    for r in con.execute("SELECT nhl_game_id, player_id, data FROM nhl_boxscore_lines"):
        lines.setdefault(r["nhl_game_id"], {})[r["player_id"]] = json.loads(r["data"])
    known = {r[0] for r in con.execute("SELECT player_id FROM players")}

    def row(g, player_id, team, role, stats, line):
        return {"game_id": g["id"], "date": g["date"], "label": g["label"], "away": g["away"], "home": g["home"],
                "is_demo": g["is_demo"], "player_id": player_id, "team": team, "role": role, "stats": stats,
                "line": line}

    out, seen = [], set()
    for r in con.execute("SELECT * FROM game_stats"):
        g = games[r["game_id"]]
        line = lines.get(g["nhl_game_id"], {}).get(r["player_id"])
        line = line if line and line.get("toi_s") else None
        out.append(row(g, r["player_id"], r["team"], r["role"], json.loads(r["stats"]), line))
        seen.add((g["id"], r["player_id"]))
    for g in games.values():
        for player_id, line in lines.get(g["nhl_game_id"], {}).items():
            if (g["id"], player_id) not in seen and player_id in known and line.get("toi_s"):
                out.append(row(g, player_id, line.get("team"), line.get("role") or "skater", None, line))
    out.sort(key=lambda r: (r["date"], r["game_id"]), reverse=True)
    return out
