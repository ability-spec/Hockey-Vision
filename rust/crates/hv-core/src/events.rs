//! `events.py`: play events from the rink positions (possessions, shots, passes, turnovers,
//! defensive plays) and per-player stats. Same rules, thresholds and order of operations as the
//! Python, so the outputs match it exactly; see events.py for what each rule means.

use std::collections::HashMap;

use crate::export::{FrameRow, PlayerRow, PuckRow, TrackRow};
use crate::pyval::{fmt_fixed, hypot, py_divmod, py_round, Obj, Val};

pub const GOAL_LINE_X: f64 = 89.0;
pub const GOAL_POST_Y: f64 = 3.0;
pub const BLUE_LINE_X: f64 = 25.0;

pub const POSSESSION_FT: f64 = 6.0;
pub const CONTESTED_FT: f64 = 1.5;
pub const CARRY_MAX_SPEED_FT_S: f64 = 45.0;
pub const VELOCITY_BASELINE_S: f64 = 0.1;
pub const MAX_VELOCITY_GAP_S: f64 = 0.3;
pub const MAX_PUCK_SPEED_FT_S: f64 = 170.0;
pub const POSSESSION_GAP_S: f64 = 1.0;
pub const MIN_TOUCHES: i64 = 3;
pub const MERGE_GAP_S: f64 = 2.0;
pub const MAX_TRANSITION_S: f64 = 3.0;
pub const MIN_PASS_FT: f64 = 10.0;
pub const STEAL_FT: f64 = 8.0;
pub const SHOT_WINDOW_S: f64 = 1.0;
pub const SHOT_MIN_SPEED_FT_S: f64 = 45.0;
pub const SHOT_MAX_DISTANCE_FT: f64 = 65.0;
pub const SHOT_NET_MARGIN_FT: f64 = 3.0;
pub const SAVE_WINDOW_S: f64 = 1.5;
pub const BLOCK_WINDOW_S: f64 = 0.5;
pub const PRESSURE_FT: f64 = 10.0;
pub const TIGHT_PRESSURE_FT: f64 = 6.0;
pub const BACKTRACK_FT: f64 = 8.0;
pub const SLOWED_FROM_FT_S: f64 = 12.0;
pub const SLOWED_RATIO: f64 = 0.5;
pub const SLOWED_WINDOW_S: f64 = 1.5;
pub const GOOD_PASS_FORWARD_FT: f64 = 15.0;
pub const OPEN_FT: f64 = 15.0;
pub const RELIEF_FT: f64 = 5.0;
pub const SMOOTH_WINDOW: usize = 5;
pub const MIN_GOALIE_X: f64 = 60.0;
pub const DIRECTION_WINDOW_S: f64 = 900.0;

const DEFENSE_PRIORITY: [&str; 3] = ["forced_pass", "forced_backtrack", "slowed"];

/// Seconds -> 'm:ss.s' video time.
pub fn clock(t: f64) -> String {
    let (div, rem) = py_divmod(t, 60.0);
    format!("{}:{:04.1}", div as i64, rem)
}

fn sign(v: f64) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

pub fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    hypot(a.0 - b.0, a.1 - b.1)
}

/// Centred moving average, the window shrinking at the ends (Python's left-to-right sum).
pub fn smooth(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let half = SMOOTH_WINDOW / 2;
    (0..points.len())
        .map(|i| {
            let win = &points[i.saturating_sub(half)..(i + half + 1).min(points.len())];
            let n = win.len() as f64;
            (
                win.iter().fold(0.0, |s, p| s + p.0) / n,
                win.iter().fold(0.0, |s, p| s + p.1) / n,
            )
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct PlayerInfo {
    pub player_id: String,
    pub team: String,
    pub team_abbrev: Option<String>,
    pub jersey_number: Option<String>,
    pub player_name: Option<String>,
    pub role: String,
    pub identified: bool,
}

impl PlayerInfo {
    pub fn to_obj(&self) -> Obj {
        Obj::new()
            .with("player_id", &self.player_id)
            .with("team", &self.team)
            .with("team_abbrev", self.team_abbrev.clone())
            .with("jersey_number", self.jersey_number.clone())
            .with("player_name", self.player_name.clone())
            .with("role", &self.role)
            .with("identified", self.identified)
    }
}

/// Players in first-seen order, looked up by id.
#[derive(Clone, Debug, Default)]
pub struct Players {
    pub list: Vec<PlayerInfo>,
    index: HashMap<String, usize>,
}

impl Players {
    pub fn get(&self, pid: &str) -> Option<usize> {
        self.index.get(pid).copied()
    }
    /// `setdefault(pid, info)`: the index of pid, adding it when new.
    pub fn setdefault(&mut self, info: PlayerInfo) -> usize {
        if let Some(&i) = self.index.get(&info.player_id) {
            return i;
        }
        self.index.insert(info.player_id.clone(), self.list.len());
        self.list.push(info);
        self.list.len() - 1
    }
}

pub type TrackKey = (i64, String);

/// Who each track is: ({(track_id, class): player index}, players). See events.identify_players.
pub fn identify_players(clip: &str, tracks: &[TrackRow]) -> (HashMap<TrackKey, usize>, Players) {
    let mut ids = HashMap::new();
    let mut players = Players::default();
    for t in tracks {
        if t.role == "referee" {
            continue;
        }
        let team_label = t
            .team_abbrev
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("team_{}", t.team.to_lowercase()));
        let pid = match t.jersey_number.as_deref().filter(|s| !s.is_empty()) {
            Some(n) => format!("{team_label} #{n}"),
            None => format!(
                "{team_label} unidentified ({clip} track {} {})",
                t.track_id, t.cls
            ),
        };
        let i = players.setdefault(PlayerInfo {
            player_id: pid,
            team: t.team.clone(),
            team_abbrev: t.team_abbrev.clone(),
            jersey_number: t.jersey_number.clone(),
            player_name: t.player_name.clone(),
            role: t.role.clone(),
            identified: t.jersey_number.as_deref().is_some_and(|s| !s.is_empty()),
        });
        ids.insert((t.track_id, t.cls.clone()), i);
        if players.list[i].player_name.is_none() && t.player_name.as_deref().is_some_and(|s| !s.is_empty()) {
            players.list[i].player_name = t.player_name.clone();
        }
    }
    (ids, players)
}

/// 'SJS #63 Zack Ostapchuk', for event descriptions.
pub fn label(p: &PlayerInfo) -> String {
    let name = match p.player_name.as_deref() {
        Some(n) if !n.is_empty() => format!(" {n}"),
        _ => String::new(),
    };
    let goalie = if p.role == "goalie" { " (G)" } else { "" };
    format!("{}{name}{goalie}", p.player_id)
}

struct Frame {
    frame: i64,
    t: f64,
    seg: i64,
    view: Option<String>,
    /// (player index, rink x, y) in first-seen order, like the Python dict.
    players: Vec<(usize, (f64, f64))>,
    puck: Option<(f64, f64)>,
}

impl Frame {
    fn get(&self, pid: usize) -> Option<(f64, f64)> {
        self.players.iter().find(|(p, _)| *p == pid).map(|(_, xy)| *xy)
    }
}

fn build_frames(
    frame_rows: &[FrameRow],
    player_rows: &[PlayerRow],
    puck_rows: &[PuckRow],
    ids: &HashMap<TrackKey, usize>,
) -> Vec<Frame> {
    let mut rows: Vec<&FrameRow> = frame_rows.iter().collect();
    rows.sort_by_key(|r| r.frame);
    let mut frames: Vec<Frame> = Vec::with_capacity(rows.len());
    let mut at: HashMap<i64, usize> = HashMap::new();
    let mut seg = 0;
    for r in rows {
        seg += r.camera_cut as i64;
        let f = Frame {
            frame: r.frame,
            t: r.time_s,
            seg,
            view: r.view.clone(),
            players: vec![],
            puck: None,
        };
        // A duplicate frame number replaces the earlier one, as in the Python dict.
        match at.get(&r.frame) {
            Some(&i) => frames[i] = f,
            None => {
                at.insert(r.frame, frames.len());
                frames.push(f);
            }
        }
    }
    for r in player_rows {
        let (Some(tid), Some(xy)) = (r.track_id, r.xy) else { continue };
        let (Some(&pid), Some(&i)) = (ids.get(&(tid, r.cls.clone())), at.get(&r.frame)) else { continue };
        if frames[i].get(pid).is_none() {
            frames[i].players.push((pid, xy));
        }
    }
    for r in puck_rows {
        if let (Some(xy), Some(&i)) = (r.xy, at.get(&r.frame)) {
            frames[i].puck = Some(xy);
        }
    }
    frames.sort_by_key(|f| f.frame);
    frames
}

/// Which way (+1 / -1) each team attacks at a given time, from where the goalies stand.
pub struct AttackDirections {
    a: Vec<(f64, i32)>,
    b: Vec<(f64, i32)>,
}

impl AttackDirections {
    fn new(frames: &[Frame], info: &Players) -> Self {
        let (mut a, mut b) = (vec![], vec![]);
        for f in frames {
            if !matches!(f.view.as_deref(), None | Some("side")) {
                continue;
            }
            for &(pid, (x, _)) in &f.players {
                let p = &info.list[pid];
                if p.role != "goalie" || x.abs() < MIN_GOALIE_X {
                    continue;
                }
                let (own, other) = if p.team == "A" { (&mut a, &mut b) } else { (&mut b, &mut a) };
                // Python's obs[p["team"]] for a team other than A / B would raise; goalies are A or B.
                own.push((f.t, -sign(x)));
                other.push((f.t, sign(x)));
            }
        }
        AttackDirections { a, b }
    }

    pub fn attack(&self, team: &str, t: f64) -> Option<i32> {
        let obs = match team {
            "A" => &self.a,
            "B" => &self.b,
            _ => return None,
        };
        let (mut total, mut n) = (0i64, 0usize);
        for &(ot, s) in obs {
            if (ot - t).abs() <= DIRECTION_WINDOW_S {
                total += s as i64;
                n += 1;
            }
        }
        if n == 0 {
            return None;
        }
        ((total.abs() as f64) >= 0.6 * n as f64).then(|| sign(total as f64))
    }
}

fn puck_velocities(frames: &[Frame]) -> HashMap<usize, (f64, f64)> {
    let mut vel = HashMap::new();
    let obs: Vec<usize> = (0..frames.len()).filter(|&i| frames[i].puck.is_some()).collect();
    for (k, &b) in obs.iter().enumerate() {
        let fb = &frames[b];
        for &a in obs[..k].iter().rev() {
            let fa = &frames[a];
            let dt = fb.t - fa.t;
            if dt > MAX_VELOCITY_GAP_S || fa.seg != fb.seg {
                break;
            }
            if dt >= VELOCITY_BASELINE_S {
                let (pb, pa) = (fb.puck.unwrap(), fa.puck.unwrap());
                vel.insert(b, ((pb.0 - pa.0) / dt, (pb.1 - pa.1) / dt));
                break;
            }
        }
    }
    vel
}

#[derive(Clone, Debug)]
struct Possession {
    pid: usize,
    start_i: usize,
    end_i: usize,
    touches: i64,
    contested: i64,
}

fn possessions(frames: &[Frame], info: &Players) -> Vec<Possession> {
    let vel = puck_velocities(frames);
    let mut touches = vec![];
    for (i, f) in frames.iter().enumerate() {
        let Some(puck) = f.puck else { continue };
        if f.players.is_empty() {
            continue;
        }
        if let Some(v) = vel.get(&i) {
            if hypot(v.0, v.1) > CARRY_MAX_SPEED_FT_S {
                continue;
            }
        }
        let mut near: Vec<(f64, usize)> = f.players.iter().map(|&(p, xy)| (dist(puck, xy), p)).collect();
        // sorted((distance, player_id)): ties go to the smaller player id string.
        near.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then_with(|| info.list[a.1].player_id.cmp(&info.list[b.1].player_id))
        });
        let (d, pid) = near[0];
        if d > POSSESSION_FT {
            continue;
        }
        let contested = near[1..]
            .iter()
            .any(|&(od, op)| od - d <= CONTESTED_FT && info.list[op].team != info.list[pid].team);
        touches.push((i, pid, contested));
    }

    let mut runs: Vec<Possession> = vec![];
    for (i, pid, contested) in touches {
        let f = &frames[i];
        if let Some(cur) = runs.last_mut() {
            if cur.pid == pid && frames[cur.end_i].seg == f.seg && f.t - frames[cur.end_i].t <= POSSESSION_GAP_S {
                cur.end_i = i;
                cur.touches += 1;
                cur.contested += contested as i64;
                continue;
            }
        }
        runs.push(Possession { pid, start_i: i, end_i: i, touches: 1, contested: contested as i64 });
    }
    runs.retain(|r| r.touches >= MIN_TOUCHES);

    let mut merged: Vec<Possession> = vec![];
    for r in runs {
        if let Some(prev) = merged.last_mut() {
            if prev.pid == r.pid
                && frames[prev.end_i].seg == frames[r.start_i].seg
                && frames[r.start_i].t - frames[prev.end_i].t <= MERGE_GAP_S
            {
                prev.end_i = r.end_i;
                prev.touches += r.touches;
                prev.contested += r.contested;
                continue;
            }
        }
        merged.push(r);
    }
    merged
}

/// (distance, player) of the closest opposing skater to `pid` in this frame.
fn nearest_opponent(frame: &Frame, pid: usize, info: &Players) -> (Option<f64>, Option<usize>) {
    let Some(me) = frame.get(pid) else { return (None, None) };
    let team = &info.list[pid].team;
    let mut best: (Option<f64>, Option<usize>) = (None, None);
    for &(op, xy) in &frame.players {
        let o = &info.list[op];
        if &o.team == team || o.role != "skater" {
            continue;
        }
        let d = dist(me, xy);
        if best.0.map_or(true, |b| d < b) {
            best = (Some(d), Some(op));
        }
    }
    best
}

fn zone(x: f64, attack: Option<i32>) -> Option<&'static str> {
    let a = x * attack? as f64;
    Some(if a > BLUE_LINE_X {
        "offensive"
    } else if a < -BLUE_LINE_X {
        "defensive"
    } else {
        "neutral"
    })
}

fn r1(v: f64) -> Val {
    Val::Float(py_round(v, 1))
}

struct Clip<'a> {
    clip: &'a str,
    frames: &'a [Frame],
    info: &'a Players,
    directions: AttackDirections,
    events: Vec<Obj>,
}

impl<'a> Clip<'a> {
    fn t(&self, i: usize) -> f64 {
        self.frames[i].t
    }

    fn name(&self, pid: usize) -> Val {
        self.info.list[pid].player_name.clone().into()
    }

    fn event(&mut self, type_: &str, subtype: &str, i: usize, pid: usize, description: String, fields: Vec<(&str, Val)>) -> usize {
        let f = &self.frames[i];
        let p = &self.info.list[pid];
        let xy = f.get(pid).or(f.puck);
        let attack = self.directions.attack(&p.team, f.t);
        let mut e = Obj::new()
            .with("clip", self.clip)
            .with("type", type_)
            .with("subtype", subtype)
            .with("frame", f.frame)
            .with("time_s", py_round(f.t, 3))
            .with("time", clock(f.t))
            .with("team", &p.team)
            .with("team_abbrev", p.team_abbrev.clone())
            .with("player_id", &p.player_id)
            .with("player_name", p.player_name.clone())
            .with("x_ft", xy.map(|v| r1(v.0)))
            .with("y_ft", xy.map(|v| r1(v.1)))
            .with("zone", xy.and_then(|v| zone(v.0, attack)));
        for (k, v) in fields {
            e.set(k, v);
        }
        e.set("description", description);
        self.events.push(e);
        self.events.len() - 1
    }

    // ---- shots

    fn shot(&mut self, p: &Possession, nxt: Option<&Possession>) -> Option<usize> {
        let frames = self.frames;
        let f_end = &frames[p.end_i];
        let release = f_end.puck.unwrap();
        let team = self.info.list[p.pid].team.clone();
        let attack = self.directions.attack(&team, f_end.t);
        let nxt_same = nxt.filter(|n| frames[n.start_i].seg == f_end.seg);
        let mut limit = f_end.t + SHOT_WINDOW_S;
        if let Some(n) = nxt_same {
            limit = limit.min(self.t(n.start_i));
        }

        // (speed, goal x, y at the goal line)
        let mut best: Option<(Option<f64>, f64, Option<f64>)> = None;
        if release.0.abs() < GOAL_LINE_X {
            let mut j = p.end_i + 1;
            while j < frames.len() && frames[j].t <= limit && frames[j].seg == f_end.seg {
                let dt = frames[j].t - f_end.t;
                if let Some(puck) = frames[j].puck.filter(|_| dt >= VELOCITY_BASELINE_S) {
                    let (vx, vy) = ((puck.0 - release.0) / dt, (puck.1 - release.1) / dt);
                    let speed = hypot(vx, vy);
                    if (SHOT_MIN_SPEED_FT_S..=MAX_PUCK_SPEED_FT_S).contains(&speed) && vx != 0.0 {
                        let goal_x = GOAL_LINE_X * sign(vx) as f64;
                        let y_cross = release.1 + vy * (goal_x - release.0) / vx;
                        if y_cross.abs() <= GOAL_POST_Y + SHOT_NET_MARGIN_FT
                            && dist(release, (goal_x, 0.0)) <= SHOT_MAX_DISTANCE_FT
                            && best.map_or(true, |b| speed > b.0.unwrap())
                        {
                            best = Some((Some(speed), goal_x, Some(y_cross)));
                        }
                    }
                }
                j += 1;
            }
        }

        // The other team's goalie getting the puck right after a release near their net.
        let mut goalie_save = false;
        if let Some(n) = nxt_same.filter(|n| self.t(n.start_i) - f_end.t <= SAVE_WINDOW_S) {
            let g = &self.info.list[n.pid];
            if let Some(gxy) = frames[n.start_i].get(n.pid) {
                if g.role == "goalie" && g.team != team && gxy.0.abs() >= MIN_GOALIE_X {
                    let goal = (GOAL_LINE_X * sign(gxy.0) as f64, 0.0);
                    if dist(release, goal) <= SHOT_MAX_DISTANCE_FT {
                        goalie_save = true;
                        if best.is_none() {
                            best = Some((None, goal.0, None));
                        }
                    }
                }
            }
        }
        let (speed, goal_x, y_cross) = best?;
        if attack.is_some_and(|a| sign(goal_x) != a) && f_end.view.as_deref() == Some("side") {
            return None; // towards their own net: a clearance or a pass back to the goalie
        }

        let mut outcome = "unknown";
        let mut by: Option<usize> = None;
        if self.puck_in_net(p.end_i, goal_x, limit + SAVE_WINDOW_S) {
            outcome = "possible_goal";
        } else if goalie_save {
            outcome = "saved";
            by = Some(nxt.unwrap().pid);
        } else if let Some(n) = nxt_same.filter(|n| self.t(n.start_i) - f_end.t <= SAVE_WINDOW_S) {
            by = Some(n.pid);
            let o = &self.info.list[n.pid];
            outcome = if o.team == team {
                "rebound_recovered"
            } else if o.role == "goalie" {
                "saved"
            } else if self.t(n.start_i) - f_end.t <= BLOCK_WINDOW_S {
                "blocked"
            } else {
                "recovered_by_opponent"
            };
        }

        let distance = dist(release, (goal_x, 0.0));
        let on_target = y_cross.map(|y| y.abs() <= GOAL_POST_Y);
        let shooter = &self.info.list[p.pid];
        let side = if goal_x > 0.0 { "right" } else { "left" };
        let mut text = format!(
            "{} shot at the {side} net from {} ft at {}",
            label(shooter),
            fmt_fixed(distance, 0),
            clock(f_end.t)
        );
        if let Some(s) = speed {
            text += &format!(", puck at {} ft/s", fmt_fixed(s, 0));
        }
        let by_label = by.map(|b| label(&self.info.list[b]));
        text += &match outcome {
            "saved" => format!("; saved by {}", by_label.as_deref().unwrap_or("the goalie")),
            "possible_goal" => "; the puck was seen in the goal mouth (possible goal)".into(),
            "blocked" => format!("; blocked by {}", by_label.as_deref().unwrap_or("a defender")),
            "rebound_recovered" => format!("; rebound recovered by teammate {}", by_label.as_deref().unwrap_or("")),
            "recovered_by_opponent" => format!("; recovered by {}", by_label.as_deref().unwrap_or("the other team")),
            _ => String::new(),
        };
        let fields = vec![
            ("distance_ft", r1(distance)),
            ("puck_speed_ft_s", speed.map(|s| py_round(s, 1)).into()),
            ("target_net", side.into()),
            ("on_target", on_target.into()),
            ("other_player_id", by.map(|b| self.info.list[b].player_id.clone()).into()),
            ("other_player_name", by.map_or(Val::None, |b| self.name(b))),
        ];
        Some(self.event("shot", outcome, p.end_i, p.pid, text, fields))
    }

    fn puck_in_net(&self, start_i: usize, goal_x: f64, until: f64) -> bool {
        let seg = self.frames[start_i].seg;
        for f in &self.frames[start_i + 1..] {
            if f.t > until || f.seg != seg {
                break;
            }
            if let Some(puck) = f.puck {
                if sign(puck.0) == sign(goal_x) && puck.0.abs() >= GOAL_LINE_X && puck.1.abs() <= GOAL_POST_Y {
                    return true;
                }
            }
        }
        false
    }

    // ---- passes and turnovers

    fn pass(&mut self, p: &Possession, nxt: &Possession, shot_next: bool) -> usize {
        let (fa, fb) = (&self.frames[p.end_i], &self.frames[nxt.start_i]);
        let (passer, receiver) = (&self.info.list[p.pid], &self.info.list[nxt.pid]);
        let attack = self.directions.attack(&passer.team, fa.t);
        let (start, end) = (fa.puck.unwrap(), fb.puck.unwrap());
        let forward = attack.map(|a| (end.0 - start.0) * a as f64);
        let (pressure, presser) = nearest_opponent(fa, p.pid, self.info);
        let (space, _) = nearest_opponent(fb, nxt.pid, self.info);

        let mut bypassed = 0i64;
        if let (Some(a), Some(fw)) = (attack, forward) {
            if fw > 0.0 {
                let a = a as f64;
                let (lo, hi) = (start.0 * a, end.0 * a);
                bypassed = fa
                    .players
                    .iter()
                    .filter(|&&(op, xy)| {
                        let o = &self.info.list[op];
                        o.team != passer.team && o.role == "skater" && lo < xy.0 * a && xy.0 * a < hi
                    })
                    .count() as i64;
            }
        }

        let mut reasons: Vec<&str> = vec![];
        if forward.is_some_and(|f| f >= GOOD_PASS_FORWARD_FT) {
            reasons.push("advanced_puck");
        }
        if bypassed > 0 {
            reasons.push("beat_defenders");
        }
        if let (Some(pr), Some(sp)) = (pressure, space) {
            if pr <= PRESSURE_FT && sp >= pr + RELIEF_FT {
                reasons.push("relieved_pressure");
            }
        }
        if space.is_some_and(|s| s >= OPEN_FT) {
            reasons.push("found_open_teammate");
        }
        if shot_next {
            reasons.push("led_to_shot");
        }

        let distance = dist(start, end);
        let mut text = format!(
            "{} passed to {} at {} ({} ft",
            label(passer),
            label(receiver),
            clock(fa.t),
            fmt_fixed(distance, 0)
        );
        if let Some(fw) = forward {
            text += &format!(
                ", {} ft {} the ice",
                fmt_fixed(fw.abs(), 0),
                if fw >= 0.0 { "up" } else { "back down" }
            );
        }
        text += ")";
        if !reasons.is_empty() {
            text += "; good pass: ";
            text += &reasons.iter().map(|r| r.replace('_', " ")).collect::<Vec<_>>().join(", ");
        }
        let pressured_by = presser.filter(|_| pressure.is_some_and(|v| v <= PRESSURE_FT));
        let fields = vec![
            ("other_player_id", receiver.player_id.clone().into()),
            ("other_player_name", receiver.player_name.clone().into()),
            ("end_time_s", py_round(fb.t, 3).into()),
            ("distance_ft", r1(distance)),
            ("forward_ft", forward.map(|v| py_round(v, 1)).into()),
            ("passer_pressure_ft", pressure.map(|v| py_round(v, 1)).into()),
            ("pressured_by", pressured_by.map(|o| self.info.list[o].player_id.clone()).into()),
            ("receiver_space_ft", space.map(|v| py_round(v, 1)).into()),
            ("opponents_bypassed", bypassed.into()),
            ("good", (!reasons.is_empty()).into()),
            ("good_reasons", Val::List(reasons.iter().map(|&r| r.into()).collect())),
        ];
        let subtype = if reasons.is_empty() { "completed" } else { "good" };
        self.event("pass", subtype, p.end_i, p.pid, text, fields)
    }

    fn turnover(&mut self, p: &Possession, nxt: &Possession) -> usize {
        let (fa, fb) = (&self.frames[p.end_i], &self.frames[nxt.start_i]);
        let (loser, gainer) = (&self.info.list[p.pid], &self.info.list[nxt.pid]);
        let gap = [fa, fb]
            .iter()
            .filter_map(|f| Some(dist(f.get(p.pid)?, f.get(nxt.pid)?)))
            .fold(None, |m: Option<f64>, d| Some(m.map_or(d, |m| if d < m { d } else { m })));
        let when = clock(fb.t);
        let (subtype, verb) = if gainer.role == "goalie" {
            ("goalie_recovery", "picked up the puck from")
        } else if gap.is_some_and(|g| g <= STEAL_FT) {
            ("takeaway", "stole the puck from")
        } else {
            ("interception", "intercepted the puck from")
        };
        let text = format!("{} {verb} {} at {when}", label(gainer), label(loser));
        let fields = vec![
            ("other_player_id", loser.player_id.clone().into()),
            ("other_player_name", loser.player_name.clone().into()),
            ("distance_between_ft", gap.map(|g| py_round(g, 1)).into()),
        ];
        self.event("turnover", subtype, nxt.start_i, nxt.pid, text, fields)
    }

    // ---- defensive plays

    fn defense(&mut self, p: &Possession, ended_by: Option<usize>) -> Option<usize> {
        let pid = p.pid;
        let team = self.info.list[pid].team.clone();
        // kind -> (defender, frame index, distance)
        let mut found: HashMap<&str, (usize, usize, f64)> = HashMap::new();

        if let Some(e) = ended_by.map(|k| &self.events[k]) {
            let (ty, sub) = (e.get("type").as_str(), e.get("subtype").as_str());
            if ty == Some("turnover") && sub == Some("takeaway") {
                return None; // reported as the takeaway
            }
            if ty == Some("pass") && e.get("pressured_by").truthy() {
                let pressure = e.get("passer_pressure_ft").as_f64().unwrap();
                let forward = e.get("forward_ft").as_f64();
                if pressure <= TIGHT_PRESSURE_FT || forward.is_some_and(|f| f <= 0.0) {
                    let defender = self.info.get(e.get("pressured_by").as_str().unwrap()).unwrap();
                    found.insert("forced_pass", (defender, p.end_i, pressure));
                }
            }
        }

        let idx: Vec<usize> = (p.start_i..=p.end_i).filter(|&i| self.frames[i].get(pid).is_some()).collect();
        if idx.len() >= 3 {
            let path = smooth(&idx.iter().map(|&i| self.frames[i].get(pid).unwrap()).collect::<Vec<_>>());
            let opp: Vec<_> = idx.iter().map(|&i| nearest_opponent(&self.frames[i], pid, self.info)).collect();
            if let Some(attack) = self.directions.attack(&team, self.t(p.start_i)) {
                let a = attack as f64;
                let mut best_k = 0;
                for k in 1..idx.len() {
                    if path[k].0 * a > path[best_k].0 * a {
                        best_k = k;
                    } else if (path[best_k].0 - path[k].0) * a >= BACKTRACK_FT {
                        // min((distance, defender id, frame)) over the opponents close enough.
                        let close = (best_k..=k)
                            .filter_map(|m| match opp[m] {
                                (Some(d), Some(op)) if d <= PRESSURE_FT => Some((d, op, idx[m])),
                                _ => None,
                            })
                            .min_by(|x, y| {
                                x.0.partial_cmp(&y.0)
                                    .unwrap()
                                    .then_with(|| self.info.list[x.1].player_id.cmp(&self.info.list[y.1].player_id))
                                    .then_with(|| x.2.cmp(&y.2))
                            });
                        if let Some((d, op, i)) = close {
                            found.insert("forced_backtrack", (op, i, d));
                        }
                        break;
                    }
                }
            }
            let speeds: Vec<Option<f64>> = (1..idx.len())
                .map(|k| {
                    let dt = self.t(idx[k]) - self.t(idx[k - 1]);
                    (dt > 0.0).then(|| dist(path[k], path[k - 1]) / dt)
                })
                .collect();
            for (a, sa) in speeds.iter().enumerate() {
                let Some(sa) = *sa else { continue };
                if sa < SLOWED_FROM_FT_S || found.contains_key("slowed") {
                    continue;
                }
                for b in a + 1..speeds.len() {
                    if self.t(idx[b + 1]) - self.t(idx[a + 1]) > SLOWED_WINDOW_S {
                        break;
                    }
                    if let (Some(sb), (Some(d), Some(op))) = (speeds[b], opp[b + 1]) {
                        if sb <= SLOWED_RATIO * sa && d <= PRESSURE_FT {
                            found.insert("slowed", (op, idx[b + 1], d));
                            break;
                        }
                    }
                }
            }
        }

        for kind in DEFENSE_PRIORITY {
            let Some(&(defender, i, d)) = found.get(kind) else { continue };
            let what = match kind {
                "forced_pass" => "forced a pass from",
                "forced_backtrack" => "forced a backtrack from",
                _ => "slowed down",
            };
            let carrier = &self.info.list[pid];
            let text = format!(
                "{} {what} {} at {} (within {} ft)",
                label(&self.info.list[defender]),
                label(carrier),
                clock(self.t(i)),
                fmt_fixed(d, 0)
            );
            let fields = vec![
                ("other_player_id", carrier.player_id.clone().into()),
                ("other_player_name", carrier.player_name.clone().into()),
                ("gap_ft", r1(d)),
            ];
            return Some(self.event("defensive_play", kind, i, defender, text, fields));
        }
        None
    }
}

pub struct ClipAnalysis {
    pub possessions: Vec<Obj>,
    pub events: Vec<Obj>,
    pub players: Players,
    /// Attacking direction per team ("A", "B") at mid-clip.
    pub directions: [Option<i32>; 2],
}

/// Possessions and play events for one clip (events.analyze_clip).
pub fn analyze_clip(
    clip: &str,
    frame_rows: &[FrameRow],
    player_rows: &[PlayerRow],
    puck_rows: &[PuckRow],
    tracks: &[TrackRow],
) -> ClipAnalysis {
    let (ids, info) = identify_players(clip, tracks);
    let frames = build_frames(frame_rows, player_rows, puck_rows, &ids);
    let mut c = Clip {
        clip,
        frames: &frames,
        info: &info,
        directions: AttackDirections::new(&frames, &info),
        events: vec![],
    };
    let poss = possessions(&frames, &info);

    let shots: Vec<Option<usize>> = (0..poss.len()).map(|k| c.shot(&poss[k], poss.get(k + 1))).collect();
    let mut ended_by: Vec<Option<usize>> = vec![None; poss.len()];
    for k in 0..poss.len().saturating_sub(1) {
        let (p, nxt) = (&poss[k], &poss[k + 1]);
        if frames[p.end_i].seg != frames[nxt.start_i].seg || frames[nxt.start_i].t - frames[p.end_i].t > MAX_TRANSITION_S {
            continue;
        }
        if shots[k].is_some() {
            continue; // what happens after a shot is its outcome (save, rebound, block)
        }
        if p.pid == nxt.pid {
            continue; // regaining one's own puck is not a pass
        }
        if info.list[p.pid].team == info.list[nxt.pid].team {
            if dist(frames[p.end_i].puck.unwrap(), frames[nxt.start_i].puck.unwrap()) >= MIN_PASS_FT {
                ended_by[k] = Some(c.pass(p, nxt, shots[k + 1].is_some()));
            }
        } else {
            ended_by[k] = Some(c.turnover(p, nxt));
        }
    }
    for (p, e) in poss.iter().zip(&ended_by) {
        c.defense(p, *e);
    }

    let mut rows = vec![];
    for (k, p) in poss.iter().enumerate() {
        let (a, b) = (&frames[p.start_i], &frames[p.end_i]);
        let pl = &info.list[p.pid];
        let end = match (shots[k], ended_by[k].map(|e| &c.events[e])) {
            (Some(_), _) => "shot".to_string(),
            (None, Some(e)) if e.get("type").as_str() == Some("turnover") => {
                format!("turnover_{}", e.get("subtype").as_str().unwrap())
            }
            (None, Some(e)) => e.get("type").as_str().unwrap().to_string(),
            (None, None) => "lost_track".to_string(),
        };
        let (pa, pb) = (a.puck.unwrap(), b.puck.unwrap());
        rows.push(
            Obj::new()
                .with("clip", clip)
                .with("player_id", &pl.player_id)
                .with("team", &pl.team)
                .with("team_abbrev", pl.team_abbrev.clone())
                .with("jersey_number", pl.jersey_number.clone())
                .with("player_name", pl.player_name.clone())
                .with("start_frame", a.frame)
                .with("end_frame", b.frame)
                .with("start_time_s", a.t)
                .with("end_time_s", b.t)
                .with("duration_s", py_round(b.t - a.t, 3))
                .with("touches", p.touches)
                .with("contested_touches", p.contested)
                .with("start_x_ft", pa.0)
                .with("start_y_ft", pa.1)
                .with("end_x_ft", pb.0)
                .with("end_y_ft", pb.1)
                .with("zone", zone(pa.0, c.directions.attack(&pl.team, a.t)))
                .with("ended_by", end),
        );
    }
    // sorted(key=(time_s, type)), stable.
    let mut events = std::mem::take(&mut c.events);
    events.sort_by(|x, y| {
        let tx = x.get("time_s").as_f64().unwrap();
        let ty = y.get("time_s").as_f64().unwrap();
        tx.partial_cmp(&ty)
            .unwrap()
            .then_with(|| x.get("type").as_str().cmp(&y.get("type").as_str()))
    });
    let mid = frames.get(frames.len() / 2).map(|f| f.t);
    let directions = ["A", "B"].map(|team| mid.and_then(|t| c.directions.attack(team, t)));
    ClipAnalysis { possessions: rows, events, players: info, directions }
}

/// One row per player (events.player_stats). `presence` holds, per player id, the Python dict
/// {"frames", "seconds", "frames_on_rink", "seconds_on_rink", "distance_ft"}.
pub fn player_stats(info: &Players, events: &[Obj], possessions: &[Obj], presence: &HashMap<String, Obj>) -> Vec<Obj> {
    let mut stats: HashMap<String, Obj> = HashMap::new();
    fn bump(stats: &mut HashMap<String, Obj>, pid: &str, key: &str, by: Val) {
        let s = stats.entry(pid.to_string()).or_default();
        let v = s.get(key).clone();
        let v = if v.is_none() { Val::Int(0) } else { v };
        s.set(key, v.add(&by));
    }
    for e in events {
        let pid = e.get("player_id").as_str().unwrap().to_string();
        let other = e.get("other_player_id").as_str().map(str::to_string);
        let sub = e.get("subtype").as_str().unwrap_or("");
        match e.get("type").as_str().unwrap_or("") {
            "shot" => {
                bump(&mut stats, &pid, "shots", Val::Int(1));
                let on_net = matches!(sub, "saved" | "possible_goal") || e.get("on_target").truthy();
                bump(&mut stats, &pid, "shots_on_net", Val::Bool(on_net));
                bump(&mut stats, &pid, "possible_goals", Val::Bool(sub == "possible_goal"));
                bump(&mut stats, &pid, "shots_blocked_by_opponent", Val::Bool(sub == "blocked"));
                if let Some(o) = other.as_deref() {
                    if sub == "saved" {
                        bump(&mut stats, o, "saves", Val::Int(1));
                    }
                    if sub == "blocked" {
                        bump(&mut stats, o, "blocked_shots", Val::Int(1));
                    }
                }
            }
            "pass" => {
                bump(&mut stats, &pid, "passes_completed", Val::Int(1));
                bump(&mut stats, &pid, "good_passes", e.get("good").clone());
                bump(&mut stats, &pid, "pass_distance_ft", e.get("distance_ft").clone());
                bump(&mut stats, other.as_deref().unwrap(), "passes_received", Val::Int(1));
            }
            "turnover" => {
                let key = match sub {
                    "takeaway" => "takeaways",
                    "interception" => "interceptions",
                    _ => "goalie_recoveries",
                };
                bump(&mut stats, &pid, key, Val::Int(1));
                bump(&mut stats, other.as_deref().unwrap(), "turnovers", Val::Int(1));
            }
            "defensive_play" => {
                bump(&mut stats, &pid, "defensive_plays", Val::Int(1));
                bump(&mut stats, &pid, &format!("defense_{sub}"), Val::Int(1));
                bump(&mut stats, other.as_deref().unwrap(), "times_defended", Val::Int(1));
            }
            _ => {}
        }
    }
    for p in possessions {
        let pid = p.get("player_id").as_str().unwrap().to_string();
        bump(&mut stats, &pid, "possessions", Val::Int(1));
        bump(&mut stats, &pid, "possession_time_s", p.get("duration_s").clone());
        bump(&mut stats, &pid, "puck_touches", p.get("touches").clone());
    }

    let empty = Obj::new();
    let mut rows: Vec<Obj> = vec![];
    for p in &info.list {
        let s = stats.get(&p.player_id).unwrap_or(&empty);
        let n = |k: &str| match s.get(k) {
            Val::None => Val::Int(0),
            v => v.clone(),
        };
        let pr = presence.get(&p.player_id).unwrap_or(&empty);
        let pget = |k: &str, default: Val| match pr.get(k) {
            Val::None if !pr.contains(k) => default,
            v => v.clone(),
        };
        let passes = n("passes_completed");
        let mean_pass = match passes.as_f64() {
            Some(c) if c != 0.0 => Val::Float(py_round(n("pass_distance_ft").as_f64().unwrap() / c, 1)),
            _ => Val::None,
        };
        let mut row = p.to_obj();
        row.set("time_detected_s", pget("seconds", Val::Float(0.0)).round(2));
        row.set("time_on_rink_s", pget("seconds_on_rink", Val::Float(0.0)).round(2));
        row.set("frames_detected", pget("frames", Val::Int(0)));
        row.set("distance_ft", pget("distance_ft", Val::Float(0.0)).round(1));
        row.set("possessions", n("possessions"));
        row.set("possession_time_s", n("possession_time_s").round(2));
        for k in [
            "puck_touches", "shots", "shots_on_net", "possible_goals", "shots_blocked_by_opponent",
            "passes_completed", "good_passes", "passes_received",
        ] {
            row.set(k, n(k));
        }
        row.set("mean_pass_distance_ft", mean_pass);
        for k in [
            "takeaways", "interceptions", "turnovers", "defensive_plays", "defense_forced_pass",
            "defense_forced_backtrack", "defense_slowed", "times_defended", "blocked_shots", "saves",
            "goalie_recoveries",
        ] {
            row.set(k, n(k));
        }
        rows.push(row);
    }
    rows.sort_by(|a, b| {
        let key = |r: &Obj| {
            let jersey = r.get("jersey_number").as_str().filter(|s| !s.is_empty()).map_or(0i64, |s| s.parse().unwrap_or(0));
            (
                !r.get("identified").truthy(),
                r.get("team").as_str().unwrap_or("").to_string(),
                r.get("team_abbrev").as_str().unwrap_or("").to_string(),
                jersey,
                r.get("player_id").as_str().unwrap_or("").to_string(),
            )
        };
        key(a).cmp(&key(b))
    });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_like_python() {
        assert_eq!(clock(0.0), "0:00.0");
        assert_eq!(clock(2.102), "0:02.1");
        assert_eq!(clock(125.55), "2:05.5");
        assert_eq!(clock(59.96), "0:60.0"); // Python's f"{59.96:04.1f}" rounds up too
    }
}
