//! run_models.py for videos: every frame through the player, puck, rink and dots models, players
//! tracked (ByteTrack), jersey numbers read off the player boxes and voted per track, camera cuts
//! found, and one record per frame written to detections.json in run_models.py's format.
//!
//! Same steps and order as the Python: the frame is downscaled once to the models' input size and
//! shared, letterboxed like Ultralytics, NMS'd, scaled back and mapped onto the frame; the number
//! model reads crops of the full-resolution frame.
//!
//! What runs at the same time (none of it changes the output):
//! - decoding and camera-cut detection, on the reader thread;
//! - with `parallel`, the four full-frame models of a frame, each on its own thread;
//! - stage 1 of frame N+1 (full-frame models, tracking) with stage 2 of frame N (jersey numbers).
//!   The jersey votes still go frame by frame, so which players are re-read doesn't change.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use hv_core::bytetrack::ByteTracker;
use hv_core::image::{letterbox_into, letterbox_params, resize_linear, Image};
use hv_core::jersey::{assemble_number, Digit, JerseyVotes};
use hv_core::pyval::{dumps, py_round, Obj, Val};
use hv_core::roster::Roster;
use hv_core::yolo::{clip_boxes, nms, scale_boxes, Det};

use crate::model::{Device, Model};
use crate::video::FrameReader;

pub const MODEL_NAMES: [&str; 5] = ["player", "puck", "number", "rink", "dots"];
const RINK_MODELS: [&str; 2] = ["rink", "dots"];

pub fn model_file(name: &str) -> &'static str {
    match name {
        "player" => "new_player_model",
        "puck" => "new_nano_puck",
        "dots" => "new_dots",
        "rink" => "new_rink_model",
        _ => "new_nums",
    }
}

pub const NUMBER_IMGSZ: usize = 640;
pub const NUMBER_BATCH: usize = 16;
pub const MIN_NUMBER_CROP_PX: i64 = 60;
pub const DIGIT_Y_RANGE: (f64, f64) = (0.1, 0.62);
pub const DIGIT_HEIGHT_RANGE: (f64, f64) = (0.06, 0.3);
pub const DEFAULT_PUCK_CONF: f32 = 0.60;

pub struct Config {
    pub models: Vec<String>,
    /// Weights per model name (.onnx).
    pub weights: HashMap<String, PathBuf>,
    pub conf: f32,
    pub puck_conf: f32,
    pub imgsz: usize,
    pub roster: Option<Roster>,
    pub jersey_stride: usize,
    pub rink_stride: usize,
    pub max_frames: usize,
    pub output: PathBuf,
    pub device: Device,
    pub threads: usize,
    /// Run the full-frame models at the same time, and overlap consecutive frames.
    pub parallel: bool,
}

/// One output box: float32 like Ultralytics' tensors, plus the track id when tracked.
#[derive(Clone, Debug)]
struct OutBox {
    xyxy: [f32; 4],
    conf: f32,
    cls: usize,
    track_id: Option<i64>,
}

struct Jersey {
    digits: Vec<Digit>,
    reading: Option<(String, f64)>,
    number: Option<String>,
}

type Results = Vec<(&'static str, Vec<OutBox>)>;

/// Time spent per stage, for --time. With `parallel`, stages overlap, so they add up to more
/// than the total.
#[derive(Default, Debug)]
pub struct Timings {
    pub frames: usize,
    pub decode_wait: Duration,
    pub resize: Duration,
    pub models: HashMap<&'static str, Duration>,
    pub tracking: Duration,
    pub numbers: Duration,
    pub write: Duration,
    pub total: Duration,
}

pub struct Detector {
    cfg: Config,
    /// Loaded full-frame models in MODEL_NAMES order.
    full: Vec<(&'static str, Model)>,
    number: Option<Model>,
    /// Class names per model as written to detections.json (the puck model's "item" is "puck").
    names: HashMap<&'static str, Vec<String>>,
}

/// (w, h) the models shrink a (w, h) frame to for input size `imgsz`, or None when they wouldn't.
fn input_size(w: usize, h: usize, imgsz: usize) -> Option<(usize, usize)> {
    let r = (imgsz as f64 / h as f64).min(imgsz as f64 / w as f64);
    if r >= 1.0 {
        return None;
    }
    Some(((w as f64 * r).round_ties_even() as usize, (h as f64 * r).round_ties_even() as usize))
}

impl Detector {
    pub fn new(mut cfg: Config) -> Result<Self, String> {
        if cfg.models.iter().any(|m| m == "number") && !cfg.models.iter().any(|m| m == "player") {
            cfg.models.push("player".into());
            eprintln!("Adding the player model: the number model needs player boxes");
        }
        let mut full = vec![];
        let mut number = None;
        let mut names = HashMap::new();
        for name in MODEL_NAMES {
            if !cfg.models.iter().any(|m| m == name) {
                continue;
            }
            let path = cfg.weights.get(name).cloned().ok_or(format!("no weights for {name}"))?;
            if !path.exists() {
                return Err(format!("Model not found: {}", path.display()));
            }
            eprintln!("  {name}: {}", path.display());
            let model = Model::load(&path, cfg.device, cfg.threads)?;
            let mut n = model.names.clone();
            if name == "puck" {
                for c in &mut n {
                    if c == "item" {
                        *c = "puck".into(); // the rest of the pipeline looks for "puck"
                    }
                }
            }
            names.insert(name, n);
            if name == "number" {
                number = Some(model);
            } else {
                full.push((name, model));
            }
        }
        Ok(Detector { cfg, full, number, names })
    }

    /// process_video: writes <output>/<video stem>/detections.json.
    pub fn process_video(&mut self, path: &Path) -> Result<Timings, String> {
        let Detector { cfg, full, number, names } = self;
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let out_dir = cfg.output.join(&stem);
        std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
        let mut video = FrameReader::open(path, cfg.max_frames)?;
        eprintln!(
            "  decoding with ffmpeg ({}{})",
            video.info.pix_fmt,
            if video.exact { ", frames identical to OpenCV's" } else { ", converted by ffmpeg: not bit-identical to OpenCV" }
        );
        let fps = video.info.fps;
        let total = if cfg.max_frames > 0 && (video.info.frames == 0 || video.info.frames > cfg.max_frames) {
            cfg.max_frames
        } else {
            video.info.frames
        };
        let mut file = BufWriter::new(File::create(out_dir.join("detections.json")).map_err(|e| e.to_string())?);
        file.write_all(b"[").map_err(|e| e.to_string())?;

        let mut tracker = ByteTracker::new();
        let mut votes: JerseyVotes<(i64, String)> = JerseyVotes::new();
        let mut t = Timings::default();
        let start = Instant::now();
        let cfg: &Config = cfg;
        let names: &HashMap<&'static str, Vec<String>> = names;

        // Stage 2 for one frame (after its stage 1): jersey numbers, then the record.
        let mut done = 0usize;
        let mut finish = |frame: Frame, number: &mut Option<Model>, t: &mut Timings| -> Result<(), String> {
            let tn = Instant::now();
            let jerseys = match number.as_mut() {
                Some(model) => Some(numbers(cfg, names, model, &mut votes, &frame)?),
                None => None,
            };
            t.numbers += tn.elapsed();
            let tw = Instant::now();
            let record = record(cfg, names, frame.idx, fps, frame.cut, frame.rink_run, &frame.results, jerseys.as_deref());
            file.write_all(if frame.idx == 0 { b"\n" } else { b",\n" }).map_err(|e| e.to_string())?;
            file.write_all(dumps(&record).as_bytes()).map_err(|e| e.to_string())?;
            t.write += tw.elapsed();
            done += 1;
            if done % 25 == 0 || done == total {
                eprint!("\r  frame {done}/{}", if total > 0 { total.to_string() } else { "?".into() });
            }
            Ok(())
        };

        let result = (|| -> Result<(), String> {
            let mut pending: Option<Frame> = None;
            let mut idx = 0usize;
            loop {
                let tw = Instant::now();
                let next = video.next().transpose()?;
                t.decode_wait += tw.elapsed();
                let Some((image, cut)) = next else { break };
                // The rink models run every rink_stride frames, and on every cut since the old fit is gone.
                let rink_run = cut || idx % cfg.rink_stride == 0;
                let this = idx;
                idx += 1;
                if !cfg.parallel {
                    let mut st = StageTimes::default();
                    let results = stage1(cfg, full, &mut tracker, &image, !rink_run, false, &mut st)?;
                    st.add_to(&mut t);
                    finish(Frame { idx: this, image, cut, rink_run, results }, number, &mut t)?;
                    continue;
                }
                // Stage 1 of this frame on another thread while stage 2 of the previous one runs here.
                let prev = pending.take();
                let (res, prev_result) = std::thread::scope(|s| {
                    let full = &mut *full;
                    let tracker = &mut tracker;
                    let image = &image;
                    let h = s.spawn(move || {
                        let mut st = StageTimes::default();
                        stage1(cfg, full, tracker, image, !rink_run, true, &mut st).map(|r| (r, st))
                    });
                    let prev_result = match prev {
                        Some(p) => finish(p, number, &mut t),
                        None => Ok(()),
                    };
                    (h.join().expect("stage 1 thread"), prev_result)
                });
                prev_result?;
                let (results, st) = res?;
                st.add_to(&mut t);
                pending = Some(Frame { idx: this, image, cut, rink_run, results });
            }
            if let Some(p) = pending.take() {
                finish(p, number, &mut t)?;
            }
            Ok(())
        })();
        // The array is closed even after an error, so the frames done so far stay usable.
        file.write_all(b"\n]\n").and_then(|_| file.flush()).map_err(|e| e.to_string())?;
        eprintln!();
        result?;
        t.frames = done;
        t.total = start.elapsed();
        Ok(t)
    }
}

struct Frame {
    idx: usize,
    image: Image,
    cut: bool,
    rink_run: bool,
    results: Results,
}

#[derive(Default)]
struct StageTimes {
    resize: Duration,
    models: Vec<(&'static str, Duration)>,
    tracking: Duration,
}

impl StageTimes {
    fn add_to(self, t: &mut Timings) {
        t.resize += self.resize;
        t.tracking += self.tracking;
        for (n, d) in self.models {
            *t.models.entry(n).or_default() += d;
        }
    }
}

/// One full-frame model on the (shared, downscaled) frame: detections in that image's pixels.
fn run_model(cfg: &Config, name: &'static str, model: &mut Model, img: &Image) -> Result<(Vec<Det>, Duration), String> {
    let conf = if name == "puck" { cfg.puck_conf } else { cfg.conf };
    let lb = letterbox_params(img.w, img.h, (cfg.imgsz, cfg.imgsz), true, model.stride);
    model.allow_static(1, lb.shape.0, lb.shape.1); // with CoreML: every frame has this shape
    let mut input = Vec::with_capacity(3 * lb.shape.0 * lb.shape.1);
    letterbox_into(img, &lb, &mut input);
    let t0 = Instant::now();
    let (shape, out) = model.run(input, 1, lb.shape.0, lb.shape.1)?;
    let took = t0.elapsed();
    // The player model's own puck class is dropped (the puck model handles pucks).
    let classes: Option<Vec<usize>> =
        (name == "player").then(|| (0..model.names.len()).filter(|&i| model.names[i] != "puck").collect());
    let mut dets = nms(&out, shape[1] - 4, shape[2], conf, classes.as_deref());
    scale_boxes(&mut dets, lb.shape, (img.h, img.w));
    Ok((dets, took))
}

/// Stage 1, run_on_frame: {model: boxes in the frame's pixels}, in MODEL_NAMES order, players tracked.
fn stage1(
    cfg: &Config,
    full: &mut [(&'static str, Model)],
    tracker: &mut ByteTracker,
    frame: &Image,
    skip_rink: bool,
    parallel: bool,
    t: &mut StageTimes,
) -> Result<Results, String> {
    let t0 = Instant::now();
    let size = input_size(frame.w, frame.h, cfg.imgsz);
    let resized_owned;
    let img = match size {
        Some((w, h)) => {
            resized_owned = resize_linear(frame, w, h);
            &resized_owned
        }
        None => frame,
    };
    t.resize += t0.elapsed();

    let todo: Vec<&mut (&'static str, Model)> =
        full.iter_mut().filter(|(n, _)| !(skip_rink && RINK_MODELS.contains(n))).collect();
    let outputs: Vec<(&'static str, Result<(Vec<Det>, Duration), String>)> = if parallel {
        std::thread::scope(|s| {
            let handles: Vec<_> = todo
                .into_iter()
                .map(|(name, model)| {
                    let name: &'static str = name;
                    (name, s.spawn(move || run_model(cfg, name, model, img)))
                })
                .collect();
            handles.into_iter().map(|(n, h)| (n, h.join().expect("model thread"))).collect()
        })
    } else {
        todo.into_iter().map(|(name, model)| (*name, run_model(cfg, name, model, img))).collect()
    };

    let mut out = vec![];
    for (name, r) in outputs {
        let (dets, took) = r?;
        t.models.push((name, took));
        let mut boxes: Vec<OutBox> =
            dets.iter().map(|d| OutBox { xyxy: d.xyxy, conf: d.conf, cls: d.cls, track_id: None }).collect();
        if name == "player" {
            let t1 = Instant::now();
            let tracks = tracker.update(&dets);
            // No active track: Ultralytics keeps the detections as they were, without ids.
            if !tracks.is_empty() {
                let mut tracked: Vec<Det> =
                    tracks.iter().map(|tr| Det { xyxy: tr.xyxy, conf: tr.score, cls: tr.cls as usize }).collect();
                clip_boxes(&mut tracked, (img.h, img.w));
                boxes = tracked
                    .iter()
                    .zip(&tracks)
                    .map(|(d, tr)| OutBox { xyxy: d.xyxy, conf: d.conf, cls: d.cls, track_id: Some(tr.track_id) })
                    .collect();
            }
            t.tracking += t1.elapsed();
        }
        if let Some((w, h)) = size {
            // to_frame_coords
            let (sx, sy) = ((frame.w as f64 / w as f64) as f32, (frame.h as f64 / h as f64) as f32);
            let mut ds: Vec<Det> = boxes
                .iter()
                .map(|b| Det { xyxy: [b.xyxy[0] * sx, b.xyxy[1] * sy, b.xyxy[2] * sx, b.xyxy[3] * sy], conf: b.conf, cls: b.cls })
                .collect();
            clip_boxes(&mut ds, (frame.h, frame.w));
            for (b, d) in boxes.iter_mut().zip(ds) {
                b.xyxy = d.xyxy;
            }
        }
        out.push((name, boxes));
    }
    Ok(out)
}

/// Stage 2: read and vote this frame's jersey numbers.
fn numbers(
    cfg: &Config,
    names: &HashMap<&'static str, Vec<String>>,
    model: &mut Model,
    votes: &mut JerseyVotes<(i64, String)>,
    frame: &Frame,
) -> Result<Vec<Jersey>, String> {
    let players = &frame.results.iter().find(|(n, _)| *n == "player").unwrap().1;
    let pnames = &names["player"];
    // Keyed on the class too, so an id handed to the other team doesn't inherit the number.
    let keys: Vec<Option<(i64, String)>> = players.iter().map(|p| p.track_id.map(|id| (id, pnames[p.cls].clone()))).collect();
    // Players whose number is settled are only read every jersey_stride frames, staggered by id.
    let only: Option<Vec<bool>> = (cfg.jersey_stride > 1).then(|| {
        keys.iter()
            .map(|k| match k {
                None => true,
                Some(key) => !votes.settled(key) || (frame.idx as i64 + key.0) % cfg.jersey_stride as i64 == 0,
            })
            .collect()
    });
    let mut js = read_jerseys(cfg, names, model, &frame.image, players, only.as_deref())?;
    for (k, j) in keys.iter().zip(&js) {
        votes.add(k.as_ref(), j.reading.as_ref());
    }
    for (k, j) in keys.iter().zip(js.iter_mut()) {
        j.number = k.as_ref().and_then(|k| votes.number(k));
    }
    Ok(js)
}

/// read_jerseys: each player's digits and reading this frame (players not in `only` aren't read).
fn read_jerseys(
    cfg: &Config,
    names: &HashMap<&'static str, Vec<String>>,
    model: &mut Model,
    frame: &Image,
    players: &[OutBox],
    only: Option<&[bool]>,
) -> Result<Vec<Jersey>, String> {
    let classes: Vec<&String> = players.iter().map(|p| &names["player"][p.cls]).collect();
    let mut jerseys: Vec<Jersey> = players.iter().map(|_| Jersey { digits: vec![], reading: None, number: None }).collect();
    let (w, h) = (frame.w as i64, frame.h as i64);
    let boxes: Vec<[f64; 4]> = players.iter().map(|p| p.xyxy.map(|v| v as f64)).collect();
    let mut crops = vec![];
    for (i, b) in boxes.iter().enumerate() {
        let (x1, y1) = (0.max(b[0] as i64), 0.max(b[1] as i64));
        let (x2, y2) = (w.min(b[2] as i64), h.min(b[3] as i64));
        if only.is_some_and(|o| !o[i]) || y2 - y1 < MIN_NUMBER_CROP_PX || x2 <= x1 {
            continue;
        }
        if classes[i] == "referee" {
            continue; // referees only get detected so they aren't taken for players
        }
        crops.push((i, x1, y1, frame.crop(x1 as usize, y1 as usize, x2 as usize, y2 as usize)));
    }
    let mut results: Vec<Vec<Det>> = vec![];
    for batch in crops.chunks(NUMBER_BATCH) {
        // Ultralytics pads to a square unless every image in the batch has the same shape.
        let same = batch.iter().all(|c| (c.3.w, c.3.h) == (batch[0].3.w, batch[0].3.h));
        let lbs: Vec<_> =
            batch.iter().map(|c| letterbox_params(c.3.w, c.3.h, (NUMBER_IMGSZ, NUMBER_IMGSZ), same, model.stride)).collect();
        let mut input = vec![];
        for (c, lb) in batch.iter().zip(&lbs) {
            letterbox_into(&c.3, lb, &mut input);
        }
        let (ih, iw) = lbs[0].shape;
        let (shape, out) = if cfg.device == Device::CoreMl && (ih, iw) == (NUMBER_IMGSZ, NUMBER_IMGSZ) {
            // CoreML: one crop at a time through the fixed-shape session (much faster than a
            // dynamic batch).
            model.allow_static(1, ih, iw);
            let per_in = 3 * ih * iw;
            let mut all = vec![];
            let mut shape = vec![];
            for k in 0..batch.len() {
                let (s1, o) = model.run(input[k * per_in..(k + 1) * per_in].to_vec(), 1, ih, iw)?;
                shape = s1;
                all.extend(o);
            }
            shape[0] = batch.len();
            (shape, all)
        } else {
            model.run(input, batch.len(), ih, iw)?
        };
        let per = shape[1] * shape[2];
        for (k, (c, lb)) in batch.iter().zip(&lbs).enumerate() {
            let mut d = nms(&out[k * per..(k + 1) * per], shape[1] - 4, shape[2], cfg.conf, None);
            scale_boxes(&mut d, lb.shape, (c.3.h, c.3.w));
            results.push(d);
        }
    }
    let digit_names = &names["number"];
    for ((i, ox, oy, _), r) in crops.iter().zip(results) {
        let (i, ox, oy) = (*i, *ox as f64, *oy as f64);
        let box_h = boxes[i][3] - boxes[i][1];
        for d in r {
            let b = d.xyxy.map(|v| v as f64);
            // Jersey numbers sit on the torso; digits elsewhere are socks, sticks or boards.
            let cyr = (b[1] + b[3]) / 2.0 / box_h;
            let hr = (b[3] - b[1]) / box_h;
            if !(DIGIT_Y_RANGE.0 <= cyr && cyr <= DIGIT_Y_RANGE.1 && DIGIT_HEIGHT_RANGE.0 <= hr && hr <= DIGIT_HEIGHT_RANGE.1) {
                continue;
            }
            // Where players overlap, a digit inside both boxes could be either one's.
            let (cx, cy) = ((b[0] + b[2]) / 2.0 + ox, (b[1] + b[3]) / 2.0 + oy);
            if boxes.iter().enumerate().any(|(k, o)| k != i && o[0] <= cx && cx <= o[2] && o[1] <= cy && cy <= o[3]) {
                continue;
            }
            jerseys[i].digits.push(Digit {
                class: digit_names[d.cls].clone(),
                confidence: py_round(d.conf as f64, 4),
                box_xyxy: [py_round(b[0] + ox, 1), py_round(b[1] + oy, 1), py_round(b[2] + ox, 1), py_round(b[3] + oy, 1)],
            });
        }
        jerseys[i].reading = assemble_number(&jerseys[i].digits);
    }
    // With --teams, a reading can only be a number the player's team wears.
    if let Some(roster) = &cfg.roster {
        for (j, cls) in jerseys.iter_mut().zip(&classes) {
            if let (Some((n, _)), Some(ok)) = (&j.reading, roster.allowed(cls)) {
                if !ok.contains(n) {
                    j.reading = None;
                }
            }
        }
    }
    Ok(jerseys)
}

/// One frame's record, as run_models.py writes it.
#[allow(clippy::too_many_arguments)]
fn record(
    cfg: &Config,
    names: &HashMap<&'static str, Vec<String>>,
    idx: usize,
    fps: f64,
    cut: bool,
    rink_run: bool,
    results: &Results,
    jerseys: Option<&[Jersey]>,
) -> Val {
    let r1 = |v: f32| Val::Float(py_round(v as f64, 1));
    let mut dets = Obj::new();
    for (name, boxes) in results {
        let mut list = vec![];
        for (i, b) in boxes.iter().enumerate() {
            let cls = &names[name][b.cls];
            let mut d = Obj::new()
                .with("class", cls.as_str())
                .with("confidence", py_round(b.conf as f64, 4))
                .with("box_xyxy", Val::List(b.xyxy.iter().map(|&v| r1(v)).collect()));
            if *name == "player" {
                if let Some(id) = b.track_id {
                    d = d.with("track_id", id);
                }
                if let Some(js) = jerseys {
                    let j = &js[i];
                    d = d.with("jersey_number", j.number.clone());
                    if let Some(roster) = &cfg.roster {
                        // The team from the class; the name once the number is known.
                        let (team, player) = roster.player(cls, j.number.as_deref());
                        d = d.with("team", team).with("player_name", player);
                    }
                    d = d.with(
                        "jersey_reading",
                        j.reading.as_ref().map(|(n, c)| Val::Obj(Obj::new().with("number", n.as_str()).with("confidence", py_round(*c, 4)))),
                    );
                }
            }
            list.push(Val::Obj(d));
        }
        dets = dets.with(name, Val::List(list));
    }
    if let Some(js) = jerseys {
        let digits: Vec<Val> = js
            .iter()
            .flat_map(|j| &j.digits)
            .map(|d| {
                Val::Obj(
                    Obj::new()
                        .with("class", d.class.as_str())
                        .with("confidence", d.confidence)
                        .with("box_xyxy", Val::List(d.box_xyxy.iter().map(|&v| Val::Float(py_round(v, 1))).collect())),
                )
            })
            .collect();
        dets = dets.with("number", Val::List(digits));
    }
    let mut rec = Obj::new().with("frame", idx).with("time_s", py_round(idx as f64 / fps, 3)).with("cut", cut);
    if !rink_run {
        rec = rec.with("rink_skipped", true); // the rink fit holds the last fit
    }
    rec.with("detections", dets).into()
}
