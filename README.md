# Hockey-Vision

Hockey-Vision turns broadcast NHL video into player and puck tracking data and play-by-play
events. It finds every player, referee and the puck in each frame, reads the players' jersey
numbers, works out where each frame's camera is looking on the rink, and maps everyone onto a
top-down rink in real-world feet. From those rink positions it finds possessions, shots, passes,
turnovers and defensive plays. The output is machine-readable data: per-frame and per-player
tables, an event list with times, and a `game_report.json` written for a language model to read
and turn into feedback.

The pipeline is one Rust program, `hockey-vision`. On an Apple M5 it runs a 600-frame 1080p60 clip
in about 37 s, against 178 s for the original Python pipeline (about 5x faster). The
exports are byte-identical. [rust/README.md](rust/README.md) has the details and the
measurements.

> The original Python pipeline (`run_models.py`, `homography.py`, `export_data.py`) is on the
> [`Python-Implementation`](../../tree/Python-Implementation) branch. That branch also has what
> the Rust pipeline doesn't do: images, `--width`, the annotated videos, training, and the Jetson /
> TensorRT setup.

## Design

```
broadcast video
      │
      ▼
hockey-vision detect ── five YOLO models per frame (ONNX Runtime) ─► outputs/<clip>/detections.json
      │   player  players, goalies (team a / b), referees; tracked across frames (ByteTrack)
      │   puck    the puck
      │   number  jersey digits on each player's box, voted per track → jersey number
      │   rink    rink keypoints (lines, circles, goals)
      │   dots    the faceoff dots on the ice
      │   (+ optional --teams roster: limits numbers to each team's and names the players)
      ▼
hockey-vision homography ── rink + dots keypoints → per-frame homography → rink feet
      │   matched to named rink landmarks, outliers dropped,     ─► positions.json/.csv,
      │   smoothed over time                                        homographies.csv
      ▼
hockey-vision export ── tables + play events ─────────────► outputs/<clip>/export/
                        possessions, shots, passes, turnovers,   *.csv, game_report.json
                        defensive plays, per-player stats
```

`hockey-vision run` does all three in one go.

1. **Detection and tracking** (`detect`): the player model's boxes are tracked across frames, so
   each player keeps an id. The number model reads digits on each player's torso, and every
   frame's reading votes for that track's number, so the number holds through frames where it
   can't be read. With `--teams`, a roster from `fetch_roster.py` restricts the numbers to those
   each team wears and names the players.
2. **Rink mapping** (`homography`): the rink and dots models' keypoints are matched to known rink
   landmarks, and a homography from image pixels to rink feet is fitted for each frame. Players
   are placed where their skates meet the ice. Rink coordinates are feet from the centre dot: x
   along the rink (-100 to 100), y across it (-42.5 to 42.5).
3. **Export and play events** (`export`): one or more clips' outputs become tables, play events
   and per-player stats (see [Output](#output)). Events are found from the rink positions alone,
   in the 2D rink plane.

## Models

The five YOLO models aren't in git. They're on Hugging Face at
[AlexRHaigh/Hockey-Vision](https://huggingface.co/AlexRHaigh/Hockey-Vision) and go in
`CV_Models/Models/`. **[CV_Models/README.md](CV_Models/README.md) lists each model, what it
detects and how to download them.** `hockey-vision` reads them as ONNX, which
`rust/export_onnx.py` makes once from the `.pt` files.

## Setup

Needs Rust 1.81 or newer ([rustup.rs](https://rustup.rs)), `ffmpeg` / `ffprobe` on the PATH, and
Python with Ultralytics once, for the ONNX export:

```bash
pip install -U huggingface_hub ultralytics
hf download AlexRHaigh/Hockey-Vision --include "*.pt" --local-dir CV_Models/Models
python rust/export_onnx.py                          # -> CV_Models/Models/onnx/*.onnx

cd rust
cargo build --release --features coreml             # on a Mac; leave out --features coreml elsewhere
cargo test --release
```

The first build downloads ONNX Runtime (via the `ort` crate).

## Running

From the repo root:

```bash
B=rust/target/release/hockey-vision
$B run videos/<clip>.mp4 --teams SJS MTL --device coreml --time   # the whole pipeline
$B detect videos/<clip>.mp4 --teams SJS MTL                         # -> outputs/<clip>/detections.json
$B homography outputs/<clip>/detections.json --frame-size 1920x1080 # -> positions.json/.csv, homographies.csv
$B export outputs/<clip>                                            # -> outputs/<clip>/export/
$B export outputs                                                   # several clips -> outputs/export/
```

`detect` and `run` take `--models`, `--conf`, `--puck-conf` (default 0.6), `--teams`, `--roster`,
`--jersey-stride` (default 5), `--rink-stride`, `--max-frames`, `--output`, `--weights NAME=PATH`,
`--device cpu|coreml`, `--threads N`, `--models-dir`, `--sequential` and `--time`.
[rust/README.md](rust/README.md#run) describes them.

`--rink-stride 2` runs the rink and dots models only on every other frame (and on every camera
cut), and the rink fit is held in between. `detect` records camera cuts in `detections.json`, so
`homography` never decodes the video.

## Output

`outputs/<clip>/export/`:

| File | Contents |
|---|---|
| `game_report.json` | Everything a language model needs for feedback, in one file: definitions, team totals, each player's stats, and every event with its time and a plain-English `description` |
| `events.csv` | One row per event: `type` (`shot`, `pass`, `turnover`, `defensive_play`), `subtype`, time, who did it, who to/from, rink position, zone; type-specific fields in `details` (JSON) |
| `player_stats.csv` | One row per player: time detected, distance skated, possessions, shots, passes (good ones too), takeaways, interceptions, turnovers, defensive plays |
| `possessions.csv` | One row per spell of a player carrying the puck, and how it ended |
| `players.csv` | One row per player detection per frame: `track_id`, `team` (A/B), `role` (skater/goalie/referee), `jersey_number`, image box, and rink `x_ft, y_ft` (empty on frames without a usable fit) |
| `puck.csv` | One row per frame with a puck: image box and rink `x_ft, y_ft` |
| `frames.csv` | One row per frame: `rink_fit`, `rejected`, `keypoints_used`, player and puck counts, `camera_cut`, `view` |
| `tracks.csv` | One row per player track: team, role, jersey number, first/last frame, frames seen, mean position, `distance_ft`, `mean_speed_ft_s` |
| `metadata.json` | Source files, frame rate, coordinate system and a description of every column |

Every table has a `clip` column, so several clips can be exported together. Each row of
`homographies.csv` is the 3x3 matrix (`h00`..`h22`) mapping a video pixel `(u, v)` to rink feet.

The events (`rust/crates/hv-core/src/events.rs` has the exact rules and thresholds):

- **Shots**: the puck leaves a player's stick at shot speed on a line at the other team's net, or
  the other team's goalie gets it straight after a release near the net. Outcomes: `saved`,
  `blocked`, `possible_goal`, `rebound_recovered`, `recovered_by_opponent`, `unknown`.
- **Passes**: the puck goes from one player to a teammate. `good` passes say why: advanced the
  puck, beat defenders, relieved pressure, found an open teammate, or led to a shot.
- **Turnovers**: the other team gets the puck: a `takeaway` (stolen off the carrier), an
  `interception`, or a `goalie_recovery`.
- **Defensive plays**: an opponent close to the carrier forces them to backtrack, slows them
  down, or makes them pass the puck away. Credited to that defender.
- **Time detected**: per player, every frame of every track carrying their jersey number.

Which way each team attacks comes from where the goalies stand. Times are video time, not the
game clock. The puck is only seen in part of the frames and players are only named once their
number is read, so treat the counts as estimates.

## Web app

`webapp/` is a web app on top of the exported data: every NHL team's players, a 0-100 score for
each player in each analysed game, favourites, playlists, and hockey pools with trade calls and an
assessment of whether to trade at all. It uses only the Python standard library (SQLite and
`http.server`), so any Python 3.9+ runs it.

```bash
python -m webapp.server                       # http://127.0.0.1:8000 (--host 0.0.0.0 to share it)
python -m webapp.ingest outputs/<clip>/export --roster rosters/2026-03-03_MTL_at_SJS.csv
python -m webapp.demo                         # or: 300 made-up demo games to try it (--clear removes them)
```

- **Overview**: your own setup rather than league totals. For your pool (or any playlist you pick)
  it shows strength, trade urgency, the average of the last game, trend and keep / watch / trade
  counts. It also charts your players' average score per week, lists your top performers and the
  players needing attention (with the best available replacement), compares your form with the
  league median by position, and lists your players' recent games and your favourites.
- **NHL data**: official, up-to-date information from the NHL's public API (documented at
  [gitlab.com/dword4/nhlapi](https://gitlab.com/dword4/nhlapi), `new-api.md`), refreshed with
  `python -m webapp.nhl sync` or the *Refresh NHL data* button on the overview:
  standings, current rosters (trades and call-ups move players; players on no roster, e.g. sent
  down or waived, are marked *Off roster*, left out of team lists and upgrade suggestions, and
  flagged in pool assessments; headshots, handedness, height, weight, birth date and place), every
  player's season totals (GP, G, A, P, +/-, PIM, PPP, SOG,
  TOI/GP; goalies W, L, OTL, GAA, SV%, SO) and this week's schedule. Player pages also fetch the
  player's last 5 games, draft and career totals when opened, cached for 6 hours. Imported games
  are matched to their NHL game by date and teams (or `webapp.ingest --game-id`) and get the
  official box score, which goes into the game scores (below). The overview adds your pool's NHL
  points, goalie wins and games this week.
- **Players**: all 32 teams from `rosters/nhl_active_players.csv` (refresh with `fetch_roster.py`;
  the server loads it into an empty database, `python -c "from webapp import db; db.load_roster(db.connect())"`
  reloads it). Each player's page has their form, trend, a chart of their game scores, their NHL
  season, and their game log.
- **Imports**: `webapp.ingest` reads an export's `player_stats.csv` and matches players to NHL ids
  by team and jersey number, so the clip must have been run with `--teams`. With `--roster` (the
  `fetch_roster.py --game` file) the game's date and teams come from the roster; otherwise pass
  `--date`, `--away` and `--home`. Importing the same folder again replaces the game.
- **Game score**: the video and the NHL's box score together, in three parts:
  - *Play*, from the video: movement, possession, pass quality and pressure (good pass +1,
    takeaway +2, turnover -2, ...), as points per minute the player was detected, blended with 3
    minutes of typical play so brief appearances can't run away.
  - *Results*, from the box score: goals +6, assists +4, shots on goal, +/-, hits, blocks, penalty
    minutes; goalies saves, goals against, the win and shutouts.
  - *Backed by video*: official results the video saw the player create in the same game count
    extra. That covers an assist when the video saw good passes, a goal when it saw the shots,
    points after it saw the player win the puck, and a plus backed by defensive plays. Turnovers on
    video in a minus game count against. Each result needs its own play on video: 4 good passes
    back at most 4 assists.

  Play is compared with every other player-game in the database, and results by where the game
  ranks among them. Until the database has enough games, results are ranked against 100 of this
  season's real box scores. The two are blended, and play counts less when the player was barely
  on screen: 50 is a typical game, 73 one spread better, 88 two. With no box score yet, the score
  is the video alone (its own shots and saves standing in); players the video never identified are
  scored on their box score alone. The matching is per game, not per play: the video's times
  aren't lined up with the game clock yet. The weights are in `webapp/scoring.py`.
- **Form and trade calls**: form is the last 5 scores, recent and longer-seen games counting more.
  A player is *Trade* below 40 form, *Watch* below 46 or after a 10-point slide, *Keep* otherwise,
  and *Hold* until 3 games are in.
- **Pools**: a playlist marked as a pool, with its league's forward / defense / goalie slots, gets
  a strength grade, depth against the slots, the best available upgrade at each weak spot, and a
  0-100 trade urgency: *hold*, *make a targeted trade*, or *be active*.

The weights and thresholds are all in `webapp/scoring.py`. Scores come from what the video picks
up, so treat them as estimates, like the counts they are built from. The database is
`webapp/hockey_vision.db` (`--db` to use another); there are no user accounts, so everyone who can
reach the server shares the favourites and playlists.

## Repository layout

| Path | Contents |
|---|---|
| `rust/crates/hv-core/` | The pipeline without the models: tracking, jersey numbers, rosters, rink fit, export and events |
| `rust/crates/hv-infer/` | The models through ONNX Runtime, the ffmpeg frame reader, the detect loop |
| `rust/crates/hv-cli/` | The `hockey-vision` binary |
| `rust/export_onnx.py` | Exports the five models to `CV_Models/Models/onnx/` |
| `rust/bench/` | Scripts that compare two runs' outputs and make synthetic test clips |
| `rust/ANALYSIS.md` | The analysis behind moving the pipeline to Rust |
| `fetch_roster.py`, `rosters/` | NHL rosters for `--teams` / `--roster` |
| `webapp/` | The web app: server, scores and trade calls, importing exports, NHL API data (`nhl.py`), the page in `static/` |
| `CV_Models/` | The models (see [CV_Models/README.md](CV_Models/README.md)) |
| `tests/` | The web app's tests (`python -m unittest discover -s tests`); the pipeline's are in `cargo test` |

## Jersey numbers

`detect` reads jersey numbers with the YOLO number model, `new_nums` (YOLO26n, classes `0`-`9`).
It detects single digits on each player's box and joins them into the number; digits outside the
torso band (stripes on socks, sticks, the boards) are ignored. Each player track's readings are
combined in a vote, so a number stays with the player through frames where it can't be read and
one-off misreads are outvoted. Referees aren't read.

### Rosters and player names

Give the two teams' abbreviations, team_a's first (the player model's `team_a_player` / `goalie_a`
classes), and jersey numbers are limited to what each team wears, players are named, and referees
get no number:

```bash
python fetch_roster.py                                   # all 32 teams' active rosters -> rosters/nhl_active_players.csv
$B run videos/<clip>.mp4 --teams SJS MTL                 # team_a = SJS, team_b = MTL
python fetch_roster.py --game 2025020969                 # a past game: exactly who dressed
$B run videos/<clip>.mp4 --teams SJS MTL --roster rosters/2026-03-03_MTL_at_SJS.csv
```

A reading that isn't a number that team wears is then dropped. `detections.json`, `positions.csv`
and the exported `players.csv` / `tracks.csv` gain `team_abbrev` / `player_name`. The roster CSV has
one row per player: `team, team_name, number, player, position, player_id, as_of`; refetch it as
rosters change. `fetch_roster.py` uses only the Python standard library.
