//! `hockey-vision`: the Hockey-Vision pipeline's command line (see rust/README.md).
//!
//!     hockey-vision homography outputs/<clip>/detections.json --frame-size 1920x1080
//!     hockey-vision export outputs/<clip> [more clips...] [--output DIR]
//!     hockey-vision detect videos/<clip>.mp4 [--teams SJS MTL] [options]
//!     hockey-vision run videos/<clip>.mp4 [--teams SJS MTL] [detect options]
//!
//! `run` is the whole pipeline in one program: detect, then the rink fit, then the export, writing
//! outputs/<clip>/detections.json, positions.*, homographies.csv and export/.
//!
//! Each command reads and writes the same files as the Python stage it replaces (on the
//! Python-Implementation branch):
//! `detect` = run_models.py (videos, without --save-video), `homography` = homography.py
//! (without --save-video), `export` = export_data.py.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use std::collections::HashMap;
use std::path::Path;

use hv_core::{detections, export, pipeline};
use hv_infer::detect::{self, Config, Detector, Timings};
use hv_infer::model::Device;

const USAGE: &str = "usage:
  hockey-vision homography <detections.json> --frame-size WxH [--output DIR] [--threads N] [--sequential] [--time]
  hockey-vision export <clip folder>... [--output DIR] [--time]
  hockey-vision detect <video or folder> [--models NAME...] [--conf X] [--puck-conf X] [--output DIR]
      [--max-frames N] [--teams TEAM_A TEAM_B] [--roster CSV] [--jersey-stride N] [--rink-stride N]
      [--weights NAME=PATH...] [--models-dir DIR] [--device cpu|coreml] [--threads N] [--sequential] [--time]
  hockey-vision run <video or folder> [detect options]   (detect + homography + export)";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let cmd = args.remove(0);
    let result = match cmd.as_str() {
        "homography" => homography(args),
        "export" => export_cmd(args),
        "detect" => detect_cmd(args),
        "run" => run_cmd(args),
        "-h" | "--help" => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        other => Err(Error::Usage(format!("unknown command {other}"))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Error::Usage(e)) => {
            eprintln!("{e}\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Error::Failed(e)) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

enum Error {
    Usage(String),
    Failed(String),
}

fn usage(e: impl Into<String>) -> Error {
    Error::Usage(e.into())
}

fn next_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Error> {
    it.next().ok_or_else(|| usage(format!("{flag} needs a value")))
}

// ---- homography

fn homography(args: Vec<String>) -> Result<(), Error> {
    let mut it = args.into_iter();
    let (mut detections_path, mut frame_size, mut output, mut threads) = (None, None, None, None);
    let (mut sequential, mut time) = (false, false);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--frame-size" => {
                let v = next_value(&mut it, "--frame-size")?;
                let (w, h) = v.split_once('x').ok_or_else(|| usage("--frame-size takes WxH, e.g. 1920x1080"))?;
                frame_size = Some((
                    w.parse::<f64>().map_err(|_| usage("bad width"))?,
                    h.parse::<f64>().map_err(|_| usage("bad height"))?,
                ));
            }
            "--output" => output = Some(PathBuf::from(next_value(&mut it, "--output")?)),
            "--threads" => {
                threads = Some(next_value(&mut it, "--threads")?.parse::<usize>().map_err(|_| usage("bad --threads"))?)
            }
            "--sequential" => sequential = true,
            "--time" => time = true,
            _ if detections_path.is_none() && !a.starts_with('-') => detections_path = Some(PathBuf::from(a)),
            _ => return Err(usage(format!("unexpected argument {a}"))),
        }
    }
    let detections_path = detections_path.ok_or_else(|| usage("homography needs a detections.json"))?;
    let frame_size = frame_size.ok_or_else(|| usage("--frame-size is required (the source video's width x height)"))?;
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().expect("thread pool");
    }

    let t0 = Instant::now();
    let frames = detections::load(&detections_path).map_err(Error::Failed)?;
    let t_load = t0.elapsed();
    let t1 = Instant::now();
    let (positions, s) = pipeline::run(&frames, frame_size, !sequential).map_err(Error::Failed)?;
    let t_fit = t1.elapsed();
    let out_dir = output.unwrap_or_else(|| detections_path.parent().map(PathBuf::from).unwrap_or_default());
    let t2 = Instant::now();
    std::fs::create_dir_all(&out_dir)
        .and_then(|_| pipeline::write_outputs(&positions, &out_dir))
        .map_err(|e| Error::Failed(format!("writing {}: {e}", out_dir.display())))?;
    let t_write = t2.elapsed();
    println!(
        "Rink fit on {}/{} frames (of the other {}: {} with clustered players, {} with implausible player size, rest too few rink markings); puck on the rink in {}",
        s.with_fit,
        s.frames,
        s.frames - s.with_fit,
        s.players_clustered,
        s.player_scale,
        s.with_puck
    );
    println!(
        "Wrote {}",
        ["positions.json", "positions.csv", "homographies.csv"]
            .map(|n| out_dir.join(n).display().to_string())
            .join(", ")
    );
    if time {
        let per_frame = t_fit.as_secs_f64() * 1e3 / s.frames.max(1) as f64;
        println!(
            "time: load {:.3}s, fit {:.3}s ({per_frame:.3} ms/frame, {} threads), write {:.3}s",
            t_load.as_secs_f64(),
            t_fit.as_secs_f64(),
            if sequential { 1 } else { rayon::current_num_threads() },
            t_write.as_secs_f64()
        );
    }
    Ok(())
}

// ---- export

fn export_cmd(args: Vec<String>) -> Result<(), Error> {
    let mut it = args.into_iter();
    let (mut paths, mut output, mut time) = (vec![], None, false);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--output" => output = Some(PathBuf::from(next_value(&mut it, "--output")?)),
            "--time" => time = true,
            _ if !a.starts_with('-') => paths.push(PathBuf::from(a)),
            _ => return Err(usage(format!("unexpected argument {a}"))),
        }
    }
    if paths.is_empty() {
        return Err(usage("export needs at least one clip folder"));
    }
    let t0 = Instant::now();
    let clips = export::find_clips(&paths);
    export::run(&clips, output.as_deref()).map_err(Error::Failed)?;
    if time {
        println!("time: export {:.3}s", t0.elapsed().as_secs_f64());
    }
    Ok(())
}

// ---- detect

const VIDEO_EXTS: [&str; 6] = ["mp4", "mov", "avi", "mkv", "m4v", "webm"];

fn parse_detect(args: Vec<String>) -> Result<(Vec<PathBuf>, Config, bool), Error> {
    let mut it = args.into_iter().peekable();
    let mut source = None;
    let mut models: Vec<String> = detect::MODEL_NAMES.iter().map(|s| s.to_string()).collect();
    let (mut conf, mut puck_conf, mut imgsz) = (0.25f32, detect::DEFAULT_PUCK_CONF, 640usize);
    let (mut output, mut max_frames) = (PathBuf::from("outputs"), 0usize);
    let (mut teams, mut roster) = (None, PathBuf::from("rosters/nhl_active_players.csv"));
    let (mut jersey_stride, mut rink_stride) = (5usize, 1usize);
    let (mut weights, mut models_dir) = (vec![], PathBuf::from("CV_Models/Models/onnx"));
    let (mut device, mut threads, mut time, mut sequential) = (Device::Cpu, 0usize, false, false);
    let num = |v: String, flag: &str| v.parse::<f64>().map_err(|_| usage(format!("bad {flag} value {v}")));
    while let Some(a) = it.next() {
        match a.as_str() {
            "--models" => {
                models.clear();
                while let Some(m) = it.next_if(|v| !v.starts_with("--")) {
                    if !detect::MODEL_NAMES.contains(&m.as_str()) {
                        return Err(usage(format!("unknown model {m}; choose from {}", detect::MODEL_NAMES.join(", "))));
                    }
                    models.push(m);
                }
            }
            "--conf" => conf = num(next_value(&mut it, &a)?, &a)? as f32,
            "--puck-conf" => puck_conf = num(next_value(&mut it, &a)?, &a)? as f32,
            "--imgsz" => imgsz = num(next_value(&mut it, &a)?, &a)? as usize,
            "--output" => output = PathBuf::from(next_value(&mut it, &a)?),
            "--max-frames" => max_frames = num(next_value(&mut it, &a)?, &a)? as usize,
            "--teams" => teams = Some([next_value(&mut it, &a)?, next_value(&mut it, &a)?]),
            "--roster" => roster = PathBuf::from(next_value(&mut it, &a)?),
            "--jersey-stride" => jersey_stride = num(next_value(&mut it, &a)?, &a)? as usize,
            "--rink-stride" => rink_stride = num(next_value(&mut it, &a)?, &a)? as usize,
            "--weights" => {
                while let Some(w) = it.next_if(|v| !v.starts_with("--")) {
                    weights.push(w);
                }
            }
            "--models-dir" => models_dir = PathBuf::from(next_value(&mut it, &a)?),
            "--device" => {
                device = match next_value(&mut it, &a)?.as_str() {
                    "cpu" => Device::Cpu,
                    "coreml" | "mps" => Device::CoreMl,
                    d => return Err(usage(format!("unknown device {d} (cpu or coreml)"))),
                }
            }
            "--threads" => threads = num(next_value(&mut it, &a)?, &a)? as usize,
            "--time" => time = true,
            "--sequential" => sequential = true,
            _ if source.is_none() && !a.starts_with('-') => source = Some(PathBuf::from(a)),
            _ => return Err(usage(format!("unexpected argument {a}"))),
        }
    }
    let source = source.ok_or_else(|| usage("detect needs a video or a folder of videos"))?;
    if !source.exists() {
        return Err(Error::Failed(format!("Source not found: {}", source.display())));
    }
    let is_video = |p: &Path| p.extension().is_some_and(|e| VIDEO_EXTS.contains(&e.to_string_lossy().to_lowercase().as_str()));
    let inputs: Vec<PathBuf> = if source.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&source)
            .map_err(|e| Error::Failed(e.to_string()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| is_video(p))
            .collect();
        v.sort();
        v
    } else if is_video(&source) {
        vec![source]
    } else {
        return Err(Error::Failed(format!("{}: only videos are supported (use the Python-Implementation branch's run_models.py for images)", source.display())));
    };
    if inputs.is_empty() {
        return Err(Error::Failed("No videos found".into()));
    }
    let mut paths: HashMap<String, PathBuf> =
        detect::MODEL_NAMES.iter().map(|n| (n.to_string(), models_dir.join(format!("{}.onnx", detect::model_file(n))))).collect();
    for spec in weights {
        let (name, path) = spec.split_once('=').ok_or_else(|| usage(format!("--weights takes NAME=PATH, got {spec}")))?;
        if !detect::MODEL_NAMES.contains(&name) {
            return Err(usage(format!("--weights: unknown model {name}")));
        }
        paths.insert(name.to_string(), PathBuf::from(path));
    }
    let roster = match teams {
        Some([a, b]) => {
            let r = hv_core::roster::Roster::load(&roster, [&a, &b]).map_err(Error::Failed)?;
            eprintln!("Rosters from {}: {}", r.path.file_name().map(|s| s.to_string_lossy()).unwrap_or_default(), r.describe());
            Some(r)
        }
        None => None,
    };
    // Default: one thread per performance core (Apple Silicon), else per core.
    let threads = if threads == 0 { default_threads() } else { threads };
    let cfg = Config {
        models,
        weights: paths,
        conf,
        puck_conf,
        imgsz,
        roster,
        jersey_stride,
        rink_stride: rink_stride.max(1),
        max_frames,
        output,
        device,
        threads,
        parallel: !sequential,
    };
    Ok((inputs, cfg, time))
}

fn print_timings(t: &Timings) {
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3 / t.frames.max(1) as f64;
    let mut models: Vec<_> = t.models.iter().collect();
    models.sort();
    println!(
        "time: {} frames in {:.2}s ({:.1} ms/frame): waiting for decode {:.1}, resize {:.1}, {}, tracking {:.2}, numbers {:.1}, writing {:.2} ms/frame",
        t.frames,
        t.total.as_secs_f64(),
        ms(t.total),
        ms(t.decode_wait),
        ms(t.resize),
        models.iter().map(|(n, d)| format!("{n} {:.1}", ms(**d))).collect::<Vec<_>>().join(", "),
        ms(t.tracking),
        ms(t.numbers),
        ms(t.write),
    );
}

fn detect_cmd(args: Vec<String>) -> Result<(), Error> {
    let (inputs, cfg, time) = parse_detect(args)?;
    let output = cfg.output.clone();
    eprintln!("Loading models: {} (device: {:?}, {} threads)", cfg.models.join(", "), cfg.device, cfg.threads);
    let mut det = Detector::new(cfg).map_err(Error::Failed)?;
    for path in inputs {
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        eprintln!("{} -> {}/", path.display(), output.join(&stem).display());
        let t = det.process_video(&path).map_err(Error::Failed)?;
        if time {
            print_timings(&t);
        }
    }
    Ok(())
}

// ---- run: the whole pipeline

fn run_cmd(args: Vec<String>) -> Result<(), Error> {
    let (inputs, cfg, time) = parse_detect(args)?;
    let output = cfg.output.clone();
    eprintln!("Loading models: {} (device: {:?}, {} threads)", cfg.models.join(", "), cfg.device, cfg.threads);
    let mut det = Detector::new(cfg).map_err(Error::Failed)?;
    for path in inputs {
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let clip_dir = output.join(&stem);
        eprintln!("{} -> {}/", path.display(), clip_dir.display());
        let t0 = Instant::now();
        let t = det.process_video(&path).map_err(Error::Failed)?;
        let t_detect = t0.elapsed();

        let t1 = Instant::now();
        let info = hv_infer::video::probe(&path).map_err(Error::Failed)?;
        let frames = detections::load(&clip_dir.join("detections.json")).map_err(Error::Failed)?;
        let (positions, s) = pipeline::run(&frames, (info.width as f64, info.height as f64), true).map_err(Error::Failed)?;
        pipeline::write_outputs(&positions, &clip_dir).map_err(|e| Error::Failed(format!("writing {}: {e}", clip_dir.display())))?;
        println!(
            "  Rink fit on {}/{} frames; puck on the rink in {}",
            s.with_fit, s.frames, s.with_puck
        );
        let t_fit = t1.elapsed();

        let t2 = Instant::now();
        export::run(&[export::py_path(&clip_dir)], None).map_err(Error::Failed)?;
        let t_export = t2.elapsed();
        if time {
            print_timings(&t);
            let n = t.frames.max(1) as f64;
            println!(
                "time per stage: detect {:.2}s ({:.1} ms/frame), rink fit {:.2}s ({:.2} ms/frame), export {:.2}s ({:.2} ms/frame)",
                t_detect.as_secs_f64(),
                t_detect.as_secs_f64() * 1e3 / n,
                t_fit.as_secs_f64(),
                t_fit.as_secs_f64() * 1e3 / n,
                t_export.as_secs_f64(),
                t_export.as_secs_f64() * 1e3 / n
            );
        }
    }
    Ok(())
}

fn default_threads() -> usize {
    let perf = std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.physicalcpu"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<usize>().ok());
    perf.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
}
