//! `export_data.py`: detections.json + positions.json -> analysis tables, play events,
//! game_report.json and metadata.json. Same files, byte for byte apart from the `generated_at`
//! timestamps.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::columns::{columns, COLUMNS, EVENT_FIELDS};
use crate::detections;
use crate::events::{self, Players, GOOD_PASS_FORWARD_FT, OPEN_FT};
use crate::pyval::{dumps, dumps_indent, fmt_fixed, hypot, py_round, utc_now_iso, Obj, Val};

pub const SMOOTH_WINDOW: usize = 5;
pub const MAX_GAP_FRAMES: i64 = 5;
pub const MAX_SPEED_FT_S: f64 = 40.0;

/// Player model classes -> (team, role).
pub fn player_role(cls: &str) -> Option<(&'static str, &'static str)> {
    Some(match cls {
        "team_a_player" => ("A", "skater"),
        "goalie_a" => ("A", "goalie"),
        "team_b_player" => ("B", "skater"),
        "goalie_b" => ("B", "goalie"),
        "referee" => ("", "referee"),
        _ => return None,
    })
}

// ---- rows

#[derive(Clone, Debug)]
pub struct PlayerRow {
    pub frame: i64,
    pub time_s: f64,
    pub track_id: Option<i64>,
    pub camera_segment: i64,
    pub cls: String,
    pub team: &'static str,
    pub role: &'static str,
    pub jersey_number: Option<String>,
    pub team_abbrev: Option<String>,
    pub player_name: Option<String>,
    pub confidence: f64,
    pub bx: [f64; 4],
    pub xy: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct PuckRow {
    pub frame: i64,
    pub time_s: f64,
    pub confidence: f64,
    pub bx: [f64; 4],
    pub xy: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct FrameRow {
    pub frame: i64,
    pub time_s: f64,
    pub camera_cut: bool,
    pub view: Option<String>,
    pub rink_fit: bool,
    pub rejected: Option<String>,
    pub keypoints_used: usize,
    pub players_detected: i64,
    pub players_on_rink: i64,
    pub puck_detected: bool,
    pub puck_on_rink: bool,
}

#[derive(Clone, Debug)]
pub struct TrackRow {
    pub track_id: i64,
    pub cls: String,
    pub team: String,
    pub role: String,
    pub jersey_number: Option<String>,
    pub team_abbrev: Option<String>,
    pub player_name: Option<String>,
    pub first_frame: i64,
    pub last_frame: i64,
    pub first_time_s: f64,
    pub last_time_s: f64,
    pub frames_detected: i64,
    pub frames_on_rink: i64,
    pub mean_x_ft: Option<f64>,
    pub mean_y_ft: Option<f64>,
    pub distance_ft: Option<f64>,
    pub mean_speed_ft_s: Option<f64>,
}

fn xy_val(xy: Option<(f64, f64)>) -> (Val, Val) {
    match xy {
        Some((x, y)) => (x.into(), y.into()),
        None => (Val::None, Val::None),
    }
}

impl TrackRow {
    fn to_obj(&self, clip: &str) -> Obj {
        Obj::new()
            .with("clip", clip)
            .with("track_id", self.track_id)
            .with("class", &self.cls)
            .with("team", &self.team)
            .with("role", &self.role)
            .with("jersey_number", self.jersey_number.clone())
            .with("team_abbrev", self.team_abbrev.clone())
            .with("player_name", self.player_name.clone())
            .with("first_frame", self.first_frame)
            .with("last_frame", self.last_frame)
            .with("first_time_s", self.first_time_s)
            .with("last_time_s", self.last_time_s)
            .with("frames_detected", self.frames_detected)
            .with("frames_on_rink", self.frames_on_rink)
            .with("mean_x_ft", self.mean_x_ft)
            .with("mean_y_ft", self.mean_y_ft)
            .with("distance_ft", self.distance_ft)
            .with("mean_speed_ft_s", self.mean_speed_ft_s)
    }
}

// ---- positions.json

#[derive(Deserialize)]
struct PosPlayer {
    #[serde(default)]
    box_xyxy: Option<[f64; 4]>,
    #[serde(default)]
    rink_xy: Option<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct PosFrame {
    frame: i64,
    #[serde(default)]
    homography: Option<serde_json::Value>,
    #[serde(default)]
    cut: Option<bool>,
    #[serde(default)]
    view: Option<String>,
    #[serde(default)]
    rejected: Option<String>,
    #[serde(default)]
    keypoints_used: Vec<serde_json::Value>,
    players: Vec<PosPlayer>,
    #[serde(default)]
    puck: Option<serde_json::Map<String, serde_json::Value>>,
}

fn rink_xy(v: &Option<Vec<Option<f64>>>) -> Option<(f64, f64)> {
    match v.as_deref() {
        Some([Some(x), Some(y), ..]) => Some((*x, *y)),
        _ => None,
    }
}

/// `tuple(round(v, 1) for v in box)` as a hash key (-0.0 and 0.0 equal, as in Python).
fn box_key(b: &[f64; 4]) -> [u64; 4] {
    b.map(|v| (py_round(v, 1) + 0.0).to_bits())
}

// ---- Python paths

/// `str(Path(s))`: no empty or "." components, no trailing slash.
pub fn py_path(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    let abs = s.starts_with('/');
    let parts: Vec<&str> = s.split('/').filter(|c| !c.is_empty() && *c != ".").collect();
    let joined = parts.join("/");
    PathBuf::from(match (abs, joined.is_empty()) {
        (true, _) => format!("/{joined}"),
        (false, true) => ".".into(),
        (false, false) => joined,
    })
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Clip folders (containing detections.json) among `paths`, or directly inside them.
pub fn find_clips(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut clips = vec![];
    for p in paths.iter().map(|p| py_path(p)) {
        if p.join("detections.json").exists() {
            clips.push(p);
        } else if p.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(&p)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .map(|e| if p == Path::new(".") { PathBuf::from(e.file_name()) } else { p.join(e.file_name()) })
                        .filter(|d| d.join("detections.json").exists())
                        .collect()
                })
                .unwrap_or_default();
            found.sort();
            if found.is_empty() {
                println!("No detections.json in {} or its subfolders, skipping", path_str(&p));
            }
            clips.extend(found);
        } else {
            println!("Not a folder: {}, skipping", path_str(&p));
        }
    }
    clips
}

// ---- one clip

pub struct ClipExport {
    pub clip: String,
    pub players: Vec<PlayerRow>,
    pub pucks: Vec<PuckRow>,
    pub frames: Vec<FrameRow>,
    pub tracks: Vec<TrackRow>,
    pub possessions: Vec<Obj>,
    pub events: Vec<Obj>,
    pub player_info: Players,
    pub presence: HashMap<String, Obj>,
    pub info: Obj,
}

/// Most common value, ties to the first seen (Counter.most_common(1)).
fn most_common<'a, T: PartialEq + 'a>(items: impl Iterator<Item = &'a T>) -> Option<&'a T> {
    let mut counts: Vec<(&T, usize)> = vec![];
    for it in items {
        match counts.iter_mut().find(|(v, _)| *v == it) {
            Some(c) => c.1 += 1,
            None => counts.push((it, 1)),
        }
    }
    let mut best: Option<(&T, usize)> = None;
    for (v, n) in counts {
        if best.map_or(true, |b| n > b.1) {
            best = Some((v, n));
        }
    }
    best.map(|b| b.0)
}

fn smooth(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    debug_assert_eq!(SMOOTH_WINDOW, events::SMOOTH_WINDOW);
    events::smooth(points)
}

/// (distance_ft, seconds it was measured over) from one track's rows, sorted by frame.
fn track_distance(rows: &[&PlayerRow]) -> (f64, f64) {
    let mut runs: Vec<Vec<&PlayerRow>> = vec![];
    let mut run: Vec<&PlayerRow> = vec![];
    for &r in rows.iter().filter(|r| r.xy.is_some()) {
        if let Some(last) = run.last() {
            if r.frame - last.frame > MAX_GAP_FRAMES || r.camera_segment != last.camera_segment {
                runs.push(std::mem::take(&mut run));
            }
        }
        run.push(r);
    }
    if !run.is_empty() {
        runs.push(run);
    }
    let (mut distance, mut seconds) = (0.0, 0.0);
    for run in runs {
        let xy = smooth(&run.iter().map(|r| r.xy.unwrap()).collect::<Vec<_>>());
        for k in 1..run.len() {
            let dt = run[k].time_s - run[k - 1].time_s;
            let step = hypot(xy[k].0 - xy[k - 1].0, xy[k].1 - xy[k - 1].1);
            if dt > 0.0 && step / dt <= MAX_SPEED_FT_S {
                distance += step;
                seconds += dt;
            }
        }
    }
    (distance, seconds)
}

pub fn summarise_tracks(players: &[PlayerRow]) -> Vec<TrackRow> {
    let mut by_track: HashMap<(i64, &str), Vec<&PlayerRow>> = HashMap::new();
    for r in players {
        if let Some(tid) = r.track_id {
            by_track.entry((tid, r.cls.as_str())).or_default().push(r);
        }
    }
    let mut keys: Vec<(i64, &str)> = by_track.keys().copied().collect();
    keys.sort();
    let mut out = vec![];
    for key in keys {
        let mut rows = by_track.remove(&key).unwrap();
        rows.sort_by_key(|r| r.frame);
        let cls = most_common(rows.iter().map(|r| &r.cls)).unwrap().clone();
        let (team, role) = player_role(&cls).unwrap();
        let number = most_common(rows.iter().filter_map(|r| r.jersey_number.as_ref().filter(|s| !s.is_empty()))).cloned();
        let named = rows
            .iter()
            .find(|r| r.jersey_number == number && r.player_name.as_deref().is_some_and(|s| !s.is_empty()));
        let abbrev = most_common(rows.iter().filter_map(|r| r.team_abbrev.as_ref().filter(|s| !s.is_empty()))).cloned();
        let on_rink: Vec<(f64, f64)> = rows.iter().filter_map(|r| r.xy).collect();
        let (distance, seconds) = track_distance(&rows);
        let n = on_rink.len() as f64;
        let mean = |f: fn(&(f64, f64)) -> f64| {
            (!on_rink.is_empty()).then(|| py_round(on_rink.iter().fold(0.0, |s, p| s + f(p)) / n, 2))
        };
        out.push(TrackRow {
            track_id: key.0,
            cls: cls.clone(),
            team: team.into(),
            role: role.into(),
            jersey_number: number,
            team_abbrev: abbrev,
            player_name: named.and_then(|r| r.player_name.clone()),
            first_frame: rows[0].frame,
            last_frame: rows[rows.len() - 1].frame,
            first_time_s: rows[0].time_s,
            last_time_s: rows[rows.len() - 1].time_s,
            frames_detected: rows.len() as i64,
            frames_on_rink: on_rink.len() as i64,
            mean_x_ft: mean(|p| p.0),
            mean_y_ft: mean(|p| p.1),
            distance_ft: (!on_rink.is_empty()).then(|| py_round(distance, 1)),
            mean_speed_ft_s: (seconds > 0.0).then(|| py_round(distance / seconds, 2)),
        });
    }
    out
}

fn fps_of(frames: &[detections::FrameRecord]) -> Option<f64> {
    // Over the whole clip: time_s is rounded to the millisecond, which skews a single frame step.
    let (first, last) = (frames.first()?, frames.last()?);
    if frames.len() < 2 || last.time_s <= first.time_s {
        return None;
    }
    Some(py_round((last.frame - first.frame) as f64 / (last.time_s - first.time_s), 3))
}

/// {player_id: frames / seconds detected and on the rink, distance skated}.
fn player_presence(clip: &str, players: &[PlayerRow], tracks: &[TrackRow], fps: Option<f64>) -> HashMap<String, Obj> {
    let (ids, info) = events::identify_players(clip, tracks);
    let mut frames: HashMap<usize, HashSet<i64>> = HashMap::new();
    let mut on_rink: HashMap<usize, HashSet<i64>> = HashMap::new();
    for r in players {
        let Some(tid) = r.track_id else { continue };
        if let Some(&pid) = ids.get(&(tid, r.cls.clone())) {
            frames.entry(pid).or_default().insert(r.frame);
            if r.xy.is_some() {
                on_rink.entry(pid).or_default().insert(r.frame);
            }
        }
    }
    let mut distance: HashMap<usize, Val> = HashMap::new();
    for t in tracks {
        if let (Some(&pid), Some(d)) = (ids.get(&(t.track_id, t.cls.clone())), t.distance_ft.filter(|d| *d != 0.0)) {
            let cur = distance.remove(&pid).unwrap_or(Val::Int(0));
            distance.insert(pid, cur.add(&Val::Float(d)));
        }
    }
    let per_frame = match fps {
        Some(f) if f != 0.0 => 1.0 / f,
        _ => 0.0,
    };
    frames
        .iter()
        .map(|(&pid, f)| {
            let n_rink = on_rink.get(&pid).map_or(0, HashSet::len);
            let p = Obj::new()
                .with("frames", f.len())
                .with("seconds", f.len() as f64 * per_frame)
                .with("frames_on_rink", n_rink)
                .with("seconds_on_rink", n_rink as f64 * per_frame)
                .with("distance_ft", distance.get(&pid).cloned().unwrap_or(Val::Int(0)));
            (info.list[pid].player_id.clone(), p)
        })
        .collect()
}

fn read_json_file(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn export_clip(clip_dir: &Path) -> Result<ClipExport, String> {
    let clip = clip_dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let det_path = clip_dir.join("detections.json");
    let frames = detections::load(&det_path)?;
    let positions_path = clip_dir.join("positions.json");
    let mut positions: HashMap<i64, PosFrame> = HashMap::new();
    if positions_path.exists() {
        let text = read_json_file(&positions_path)?;
        let list: Vec<PosFrame> = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", positions_path.display()))?;
        if list.iter().any(|p| p.players.iter().any(|pl| pl.box_xyxy.is_none())) {
            return Err(format!(
                "{} is from an older homography.py without player boxes; re-run homography.py and export again",
                path_str(&positions_path)
            ));
        }
        positions = list.into_iter().map(|p| (p.frame, p)).collect();
    } else {
        println!("  {clip}: no positions.json, exporting without rink coordinates (run homography.py first to get them)");
    }

    let (mut players, mut pucks, mut frame_rows) = (vec![], vec![], vec![]);
    let mut camera_segment = 0;
    for f in &frames {
        let pos = positions.get(&f.frame);
        let cut = pos.and_then(|p| p.cut).unwrap_or(false);
        camera_segment += cut as i64;
        let on_rink: HashMap<[u64; 4], &PosPlayer> = pos
            .map(|p| p.players.iter().map(|pl| (box_key(pl.box_xyxy.as_ref().unwrap()), pl)).collect())
            .unwrap_or_default();

        let (mut n_players, mut n_on_rink) = (0, 0);
        for d in f.model("player") {
            let Some((team, role)) = player_role(&d.cls) else { continue };
            let xy = on_rink.get(&box_key(&d.box_xyxy)).and_then(|pl| rink_xy(&pl.rink_xy));
            n_players += 1;
            n_on_rink += xy.is_some() as i64;
            players.push(PlayerRow {
                frame: f.frame,
                time_s: f.time_s,
                track_id: d.track_id,
                camera_segment,
                cls: d.cls.clone(),
                team,
                role,
                jersey_number: d.jersey_number.clone(),
                team_abbrev: d.team.clone(),
                player_name: d.player_name.clone(),
                confidence: d.confidence,
                bx: d.box_xyxy,
                xy,
            });
        }

        // Same choice as homography.project_puck: the most confident puck is the puck.
        let mut best: Option<&detections::Detection> = None;
        for d in f.model("puck").iter().filter(|d| d.cls == "puck") {
            if best.map_or(true, |b| d.confidence > b.confidence) {
                best = Some(d);
            }
        }
        let puck = pos.and_then(|p| p.puck.as_ref()).filter(|m| !m.is_empty());
        if let Some(b) = best {
            let xy = puck.and_then(|m| {
                let v: Option<Vec<Option<f64>>> = serde_json::from_value(m.get("rink_xy")?.clone()).ok()?;
                rink_xy(&v)
            });
            pucks.push(PuckRow { frame: f.frame, time_s: f.time_s, confidence: b.confidence, bx: b.box_xyxy, xy });
        }

        frame_rows.push(FrameRow {
            frame: f.frame,
            time_s: f.time_s,
            camera_cut: cut,
            view: pos.and_then(|p| p.view.clone()),
            rink_fit: pos.is_some_and(|p| p.homography.as_ref().is_some_and(|h| !h.is_null())),
            rejected: pos.and_then(|p| p.rejected.clone()),
            keypoints_used: pos.map_or(0, |p| p.keypoints_used.len()),
            players_detected: n_players,
            players_on_rink: n_on_rink,
            puck_detected: best.is_some(),
            puck_on_rink: puck.is_some(),
        });
    }

    let tracks = summarise_tracks(&players);
    let (mut possessions, mut evs, mut directions) = (vec![], vec![], [None, None]);
    let (_, mut player_info) = events::identify_players(&clip, &tracks);
    if !positions.is_empty() {
        let a = events::analyze_clip(&clip, &frame_rows, &players, &pucks, &tracks);
        (possessions, evs, player_info, directions) = (a.possessions, a.events, a.players, a.directions);
    }
    for (n, e) in evs.iter_mut().enumerate() {
        e.set("event_id", format!("{clip}-{}", n + 1));
    }

    let fps = fps_of(&frames);
    let dir_name = |d: Option<i32>| match d {
        Some(1) => "right (+x)",
        Some(-1) => "left (-x)",
        _ => "unknown",
    };
    let info = Obj::new()
        .with("clip", clip.as_str())
        .with(
            "source",
            Obj::new()
                .with("detections", path_str(&det_path))
                .with("positions", (!positions.is_empty()).then(|| path_str(&positions_path))),
        )
        .with("frames", frames.len())
        .with("fps", fps)
        .with("frames_with_rink_fit", frame_rows.iter().filter(|r| r.rink_fit).count())
        .with("duration_s", frames.last().map_or(0.0, |f| f.time_s))
        .with(
            "attacking_direction",
            Obj::new().with("A", dir_name(directions[0])).with("B", dir_name(directions[1])),
        );
    let presence = player_presence(&clip, &players, &tracks, fps);
    Ok(ClipExport {
        clip,
        players,
        pucks,
        frames: frame_rows,
        tracks,
        possessions,
        events: evs,
        player_info,
        presence,
        info,
    })
}

// ---- several clips

fn merge_presence(total: &mut HashMap<String, Obj>, presence: &HashMap<String, Obj>) {
    for (pid, p) in presence {
        let t = total.entry(pid.clone()).or_default();
        for (k, v) in &p.0 {
            let cur = match t.get(k) {
                Val::None => Val::Int(0),
                c => c.clone(),
            };
            t.set(k, cur.add(v));
        }
    }
}

const TEAM_TOTAL_PREFIXES: [&str; 12] = [
    "shots", "possible", "passes", "good", "takeaways", "interceptions", "turnovers", "defens", "defense",
    "blocked", "saves", "possession",
];

/// Summed player stats per team.
fn team_totals(stats: &[Obj]) -> Obj {
    let keys: Vec<&str> = columns("player_stats.csv")
        .into_iter()
        .filter(|k| TEAM_TOTAL_PREFIXES.iter().any(|p| k.starts_with(p)))
        .collect();
    let mut out = Obj::new();
    for r in stats {
        let team = match r.get("team_abbrev").as_str().filter(|s| !s.is_empty()) {
            Some(t) => t.to_string(),
            None => format!("team_{}", r.get("team").as_str().unwrap_or("").to_lowercase()),
        };
        if !out.contains(&team) {
            let mut t = Obj::new()
                .with("team", r.get("team"))
                .with("team_abbrev", r.get("team_abbrev"))
                .with("players_identified", 0i64);
            for k in &keys {
                t.set(k, 0i64);
            }
            out.set(&team, t);
        }
        let Some((_, Val::Obj(t))) = out.0.iter_mut().find(|(k, _)| *k == team) else { unreachable!() };
        t.set("players_identified", t.get("players_identified").add(r.get("identified")));
        for k in &keys {
            let v = r.get(k);
            let v = if v.truthy() { v.clone() } else { Val::Int(0) };
            t.set(k, t.get(k).add(&v));
        }
    }
    for (_, t) in out.0.iter_mut() {
        if let Val::Obj(t) = t {
            t.set("possession_time_s", t.get("possession_time_s").round(2));
        }
    }
    out
}

fn game_report(infos: &[Obj], stats: &[Obj], events: &[Obj]) -> Val {
    let event_types = Obj::new()
        .with("shot", "the puck left a player's stick fast towards the other team's net, or their goalie got it straight after a release near the net")
        .with("pass", "the puck went from one player to a teammate; 'good' passes list why in good_reasons")
        .with("turnover", "the other team got the puck: takeaway = stolen off the carrier, interception = picked off a pass or loose puck, goalie_recovery = their goalie picked it up")
        .with("defensive_play", "an opponent within 10 ft made the carrier backtrack, slow down or pass the puck away; credited to that defender (stealing the puck is a turnover/takeaway)");
    let reasons = Obj::new()
        .with("advanced_puck", format!("moved the puck at least {} ft up the ice", fmt_fixed(GOOD_PASS_FORWARD_FT, 0)))
        .with("beat_defenders", "the puck went past opposing skaters")
        .with("relieved_pressure", "the passer was pressured and the receiver had more space")
        .with("found_open_teammate", format!("no opponent within {} ft of the receiver", fmt_fixed(OPEN_FT, 0)))
        .with("led_to_shot", "the receiver shot");
    let definitions = Obj::new()
        .with("time", "video time (m:ss.s from the start of each clip), not the game clock")
        .with("rink", "feet from centre ice; x along the rink (-100 left end, +100 right end), y across it")
        .with("zone", "offensive / neutral / defensive for the team of the player named in the event")
        .with("player_id", "team and jersey number; 'unidentified' players are tracks whose number was never read (listed only when they took part in an event)")
        .with("event_types", event_types)
        .with("good_pass_reasons", reasons)
        .with("accuracy", "found automatically from broadcast video: the puck is seen in only part of the frames, players are only identified once their number is read, and replays can repeat plays. Treat counts as estimates.");
    let mut in_events: HashSet<Option<&str>> = events.iter().map(|e| e.get("player_id").as_str()).collect();
    in_events.extend(events.iter().map(|e| e.get("other_player_id").as_str()));
    let players: Vec<Val> = stats
        .iter()
        .filter(|r| r.get("identified").truthy() || in_events.contains(&r.get("player_id").as_str()))
        .map(|r| Val::Obj(r.clone()))
        .collect();
    Obj::new()
        .with("generated_at", utc_now_iso())
        .with("clips", Val::List(infos.iter().cloned().map(Val::Obj).collect()))
        .with("definitions", definitions)
        .with("teams", team_totals(stats))
        .with("players", players)
        .with("events", Val::List(events.iter().cloned().map(Val::Obj).collect()))
        .into()
}

/// A table's rows: dicts, or typed rows that write their cells straight in column order.
enum Rows {
    Dicts(Vec<Obj>),
    Players(Vec<(usize, PlayerRow)>),
    Pucks(Vec<(usize, PuckRow)>),
    Frames(Vec<(usize, FrameRow)>),
}

fn write_table(path: &Path, cols: &[&str], rows: &Rows, clips: &[String]) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(cols.join(",").as_bytes())?;
    f.write_all(b"\r\n")?;
    let mut line = String::new();
    let mut emit = |f: &mut BufWriter<File>, cells: &mut dyn Iterator<Item = Val>| -> std::io::Result<()> {
        line.clear();
        for (k, v) in cells.enumerate() {
            if k > 0 {
                line.push(',');
            }
            line.push_str(&v.csv());
        }
        line.push_str("\r\n");
        f.write_all(line.as_bytes())
    };
    match rows {
        Rows::Dicts(rows) => {
            for r in rows {
                emit(&mut f, &mut cols.iter().map(|c| r.get(c).clone()))?;
            }
        }
        Rows::Players(rows) => {
            // The biggest table by far: written straight into one line buffer.
            debug_assert_eq!(cols, PLAYER_COLS);
            let mut line = String::with_capacity(256);
            for (c, r) in rows {
                line.clear();
                r.write_csv(&clips[*c], &mut line);
                f.write_all(line.as_bytes())?;
            }
        }
        Rows::Pucks(rows) => {
            debug_assert_eq!(cols, PUCK_COLS);
            for (c, r) in rows {
                emit(&mut f, &mut r.cells(&clips[*c]).into_iter())?;
            }
        }
        Rows::Frames(rows) => {
            debug_assert_eq!(cols, FRAME_COLS);
            for (c, r) in rows {
                emit(&mut f, &mut r.cells(&clips[*c]).into_iter())?;
            }
        }
    }
    f.flush()
}

const PLAYER_COLS: &[&str] = &[
    "clip", "frame", "time_s", "track_id", "class", "team", "role", "jersey_number", "team_abbrev",
    "player_name", "confidence", "box_x1", "box_y1", "box_x2", "box_y2", "x_ft", "y_ft",
];
const PUCK_COLS: &[&str] = &[
    "clip", "frame", "time_s", "confidence", "box_x1", "box_y1", "box_x2", "box_y2", "x_ft", "y_ft",
];
const FRAME_COLS: &[&str] = &[
    "clip", "frame", "time_s", "rink_fit", "rejected", "keypoints_used", "players_detected",
    "players_on_rink", "puck_detected", "puck_on_rink", "camera_cut", "view",
];

impl PlayerRow {
    fn write_csv(&self, clip: &str, out: &mut String) {
        use crate::pyfmt::{push_csv_field as s, push_py_float as fl};
        use std::fmt::Write;
        let opt = |out: &mut String, v: &Option<String>| {
            if let Some(v) = v {
                s(out, v)
            }
        };
        s(out, clip);
        write!(out, ",{},", self.frame).unwrap();
        fl(out, self.time_s);
        out.push(',');
        if let Some(t) = self.track_id {
            write!(out, "{t}").unwrap();
        }
        out.push(',');
        s(out, &self.cls);
        out.push(',');
        out.push_str(self.team);
        out.push(',');
        out.push_str(self.role);
        out.push(',');
        opt(out, &self.jersey_number);
        out.push(',');
        opt(out, &self.team_abbrev);
        out.push(',');
        opt(out, &self.player_name);
        out.push(',');
        fl(out, self.confidence);
        for v in self.bx {
            out.push(',');
            fl(out, v);
        }
        out.push(',');
        if let Some((x, y)) = self.xy {
            fl(out, x);
            out.push(',');
            fl(out, y);
        } else {
            out.push(',');
        }
        out.push_str("\r\n");
    }

    #[cfg(test)]
    fn cells(&self, clip: &str) -> [Val; 17] {
        let (x, y) = xy_val(self.xy);
        [
            clip.into(), self.frame.into(), self.time_s.into(), self.track_id.into(), (&self.cls).into(),
            self.team.into(), self.role.into(), self.jersey_number.clone().into(), self.team_abbrev.clone().into(),
            self.player_name.clone().into(), self.confidence.into(), self.bx[0].into(), self.bx[1].into(),
            self.bx[2].into(), self.bx[3].into(), x, y,
        ]
    }
}

impl PuckRow {
    fn cells(&self, clip: &str) -> [Val; 10] {
        let (x, y) = xy_val(self.xy);
        [
            clip.into(), self.frame.into(), self.time_s.into(), self.confidence.into(), self.bx[0].into(),
            self.bx[1].into(), self.bx[2].into(), self.bx[3].into(), x, y,
        ]
    }
}

impl FrameRow {
    fn cells(&self, clip: &str) -> [Val; 12] {
        [
            clip.into(), self.frame.into(), self.time_s.into(), (self.rink_fit as i64).into(),
            self.rejected.clone().into(), self.keypoints_used.into(), self.players_detected.into(),
            self.players_on_rink.into(), (self.puck_detected as i64).into(), (self.puck_on_rink as i64).into(),
            (self.camera_cut as i64).into(), self.view.clone().into(),
        ]
    }
}

pub struct ExportSummary {
    pub out_dir: PathBuf,
}

/// export_data.py's main(): export `clip_dirs` (already found with find_clips) into `output`
/// (default: <clip>/export for one clip, outputs/export for several).
pub fn run(clips: &[PathBuf], output: Option<&Path>) -> Result<ExportSummary, String> {
    if clips.is_empty() {
        return Err("No clip folders to export".into());
    }
    let out_dir = match output {
        Some(o) => py_path(o),
        None if clips.len() == 1 => clips[0].join("export"),
        None => PathBuf::from("outputs/export"),
    };
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;

    let mut tables: HashMap<&str, Vec<Obj>> = HashMap::new();
    let (mut players, mut pucks, mut frames, mut clip_names) = (vec![], vec![], vec![], vec![]);
    let (mut infos, mut all_events) = (vec![], vec![]);
    let mut player_info = Players::default();
    let mut presence: HashMap<String, Obj> = HashMap::new();
    for clip_dir in clips {
        let c = export_clip(clip_dir)?;
        let clip = c.clip.as_str();
        let ci = clip_names.len();
        clip_names.push(c.clip.clone());
        players.extend(c.players.iter().map(|r| (ci, r.clone())));
        pucks.extend(c.pucks.iter().map(|r| (ci, r.clone())));
        frames.extend(c.frames.iter().map(|r| (ci, r.clone())));
        tables.entry("tracks.csv").or_default().extend(c.tracks.iter().map(|r| r.to_obj(clip)));
        tables.entry("possessions.csv").or_default().extend(c.possessions.iter().cloned());
        let count = |t: &str| c.events.iter().filter(|e| e.get("type").as_str() == Some(t)).count();
        println!(
            "  {}: {} frames, {} with a rink fit, {} player tracks, {} possessions, {} shot, {} pass, {} turnover, {} defensive_play events",
            c.clip,
            c.info.get("frames").csv(),
            c.info.get("frames_with_rink_fit").csv(),
            c.tracks.len(),
            c.possessions.len(),
            count("shot"),
            count("pass"),
            count("turnover"),
            count("defensive_play")
        );
        all_events.extend(c.events.iter().cloned());
        for p in &c.player_info.list {
            player_info.setdefault(p.clone());
        }
        merge_presence(&mut presence, &c.presence);
        infos.push(c.info);
    }

    let possessions = tables.remove("possessions.csv").unwrap_or_default();
    let stats = events::player_stats(&player_info, &all_events, &possessions, &presence);
    tables.insert("possessions.csv", possessions);
    let event_rows: Vec<Obj> = all_events
        .iter()
        .map(|e| {
            let mut row = Obj::new();
            for k in columns("events.csv") {
                if k != "details" {
                    row.set(k, e.get(k));
                }
            }
            let mut details = Obj::new();
            for (k, v) in &e.0 {
                if !EVENT_FIELDS.contains(&k.as_str()) && k != "event_id" {
                    details.set(k, v);
                }
            }
            row.set("details", dumps(&Val::Obj(details)));
            row
        })
        .collect();
    tables.insert("player_stats.csv", stats.clone());
    tables.insert("events.csv", event_rows);

    let io = |e: std::io::Error| format!("writing {}: {e}", out_dir.display());
    let mut typed = [Some(Rows::Players(players)), Some(Rows::Pucks(pucks)), Some(Rows::Frames(frames))];
    for (name, _) in COLUMNS {
        let cols = columns(name);
        let rows = match *name {
            "players.csv" => typed[0].take().unwrap(),
            "puck.csv" => typed[1].take().unwrap(),
            "frames.csv" => typed[2].take().unwrap(),
            _ => Rows::Dicts(tables.remove(name).unwrap_or_default()),
        };
        write_table(&out_dir.join(name), &cols, &rows, &clip_names).map_err(io)?;
    }
    std::fs::write(out_dir.join("game_report.json"), dumps_indent(&game_report(&infos, &stats, &all_events), 2)).map_err(io)?;

    let tables_meta: Obj = Obj(COLUMNS
        .iter()
        .map(|(t, cols)| (t.to_string(), Val::Obj(Obj(cols.iter().map(|(c, d)| (c.to_string(), Val::from(*d))).collect()))))
        .collect());
    let metadata: Val = Obj::new()
        .with("generated_at", utc_now_iso())
        .with("clips", Val::List(infos.into_iter().map(Val::Obj).collect()))
        .with(
            "coordinates",
            Obj::new()
                .with("units", "feet")
                .with("origin", "centre ice dot")
                .with("x", "along the rink, -100 (left end boards) to 100 (right end boards)")
                .with("y", "across the rink, -42.5 (near boards, camera side) to 42.5 (far boards)")
                .with("player_point", "bottom-centre of the player's box (skates on the ice)"),
        )
        .with(
            "notes",
            Obj::new()
                .with(
                    "distance",
                    format!(
                        "positions smoothed with a {SMOOTH_WINDOW}-frame centred moving average; no distance across camera cuts or gaps over {MAX_GAP_FRAMES} frames; steps over {} ft/s dropped",
                        crate::pyfmt::py_float(MAX_SPEED_FT_S)
                    ),
                )
                .with("track_id", "unique within a clip only; tracks.csv groups by (track_id, class), matching jersey voting. Same-class ID handovers can still mix players in a scrum")
                .with("events", "found from rink positions only; see events.py for the rules and thresholds"),
        )
        .with("tables", tables_meta)
        .into();
    std::fs::write(out_dir.join("metadata.json"), dumps_indent(&metadata, 2)).map_err(io)?;
    println!(
        "Wrote {}, game_report.json and metadata.json to {}/",
        COLUMNS.iter().map(|(t, _)| *t).collect::<Vec<_>>().join(", "),
        path_str(&out_dir)
    );
    Ok(ExportSummary { out_dir })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_columns_match_python() {
        assert_eq!(columns("players.csv"), PLAYER_COLS);
        assert_eq!(columns("puck.csv"), PUCK_COLS);
        assert_eq!(columns("frames.csv"), FRAME_COLS);
    }

    #[test]
    fn player_row_fast_writer_matches_cells() {
        let r = PlayerRow {
            frame: 3, time_s: 0.05, track_id: Some(7), camera_segment: 0, cls: "goalie_a".into(),
            team: "A", role: "goalie", jersey_number: Some("31".into()), team_abbrev: None,
            player_name: Some("O'Neil, Jr.".into()), confidence: 0.9, bx: [1.0, 2.5, 1e-5, 300.0], xy: None,
        };
        let mut line = String::new();
        r.write_csv("c", &mut line);
        let cells: Vec<String> = r.cells("c").iter().map(Val::csv).collect();
        assert_eq!(line, cells.join(",") + "\r\n");
    }
}
