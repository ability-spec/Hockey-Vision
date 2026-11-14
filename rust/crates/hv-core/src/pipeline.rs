//! `homography.py`'s main loop over a clip, and its outputs (positions.json, positions.csv,
//! homographies.csv), without the optional videos.
//!
//! Matching landmarks and fitting each frame's homography only depend on that frame, so they run
//! in parallel across frames (rayon); the temporal part (the tracker, rejecting bad fits) then runs
//! in frame order, as in Python. The result is the same either way.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use rayon::prelude::*;
use serde::Serialize;

use crate::detections::FrameRecord;
use crate::geometry::Mat3;
use crate::homography::{
    compute_homography, implied_player_height, match_keypoints, players_clustered, project_players,
    project_puck, FrameSize, HomographyTracker, PlayerPosition, PuckPosition, View,
    PLAYER_HEIGHT_RANGE_FT,
};
use crate::pyfmt::{csv_field, py_float};
use crate::rink::{keypoints, KeypointId};

#[derive(Clone, Debug, Serialize)]
pub struct Position {
    pub frame: i64,
    pub time_s: f64,
    pub homography: Option<[[f64; 3]; 3]>,
    pub cut: bool,
    pub view: &'static str,
    pub rejected: Option<&'static str>,
    pub keypoints_used: Vec<String>,
    pub players: Vec<PlayerPosition>,
    pub puck: Option<PuckPosition>,
}

#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub frames: usize,
    pub with_fit: usize,
    pub players_clustered: usize,
    pub player_scale: usize,
    pub with_puck: usize,
}

/// One frame's own fit, before smoothing: (H, keypoints the fit used, camera view).
pub struct FrameFit {
    pub h: Option<Mat3>,
    pub inliers: Vec<KeypointId>,
    pub view: View,
}

pub fn fit_frame(frame: &FrameRecord, fs: FrameSize) -> FrameFit {
    let (matches, lines, view) = match_keypoints(frame, fs);
    let (h, inliers) = compute_homography(&matches, &lines, fs);
    FrameFit {
        h,
        inliers: inliers.iter().map(|m| m.keypoint).collect(),
        view,
    }
}

fn round8(h: &Mat3) -> [[f64; 3]; 3] {
    // np.round(H, 8): scale, round half to even, scale back.
    std::array::from_fn(|r| std::array::from_fn(|c| (h[(r, c)] * 1e8).round_ties_even() / 1e8))
}

/// Map every frame onto the rink. `parallel` fits the frames on rayon's thread pool.
pub fn run(
    frames: &[FrameRecord],
    fs: FrameSize,
    parallel: bool,
) -> Result<(Vec<Position>, Summary), String> {
    if let Some(f) = frames.iter().find(|f| f.cut.is_none()) {
        return Err(format!(
            "frame {} has no camera cut recorded; re-run run_models.py (or use homography.py, which decodes the video to find cuts)",
            f.frame
        ));
    }
    let fit = |f: &FrameRecord| (!f.rink_skipped).then(|| fit_frame(f, fs));
    let fits: Vec<Option<FrameFit>> = if parallel {
        frames.par_iter().map(fit).collect()
    } else {
        frames.iter().map(fit).collect()
    };

    let k = keypoints();
    let mut tracker = HomographyTracker::default();
    let mut view = View::Side;
    let mut out = Vec::with_capacity(frames.len());
    let mut summary = Summary {
        frames: frames.len(),
        ..Default::default()
    };
    for (f, own) in frames.iter().zip(fits) {
        let cut = f.cut.unwrap_or(false);
        if cut {
            tracker.reset();
        }
        let (mut h, mut inliers) = match own {
            // run_models.py --rink-stride didn't run the rink models: keep the last fit.
            None => (tracker.h, vec![]),
            Some(fit) => {
                view = fit.view;
                (tracker.update(fit.h, fs), fit.inliers)
            }
        };
        let mut players = project_players(f, h.as_ref());
        let mut puck = project_puck(f, h.as_ref());

        let height = implied_player_height(f, h.as_ref());
        let rejected = if players_clustered(&players) {
            Some("players_clustered")
        } else if height
            .is_some_and(|v| !(PLAYER_HEIGHT_RANGE_FT.0 <= v && v <= PLAYER_HEIGHT_RANGE_FT.1))
        {
            Some("player_scale")
        } else {
            None
        };
        if let Some(r) = rejected {
            // The fit is wrong: drop this frame's output and don't let the tracker carry it on.
            (h, players, puck, inliers) = (None, vec![], None, vec![]);
            tracker.reset();
            if r == "players_clustered" {
                summary.players_clustered += 1;
            } else {
                summary.player_scale += 1;
            }
        }
        summary.with_fit += h.is_some() as usize;
        summary.with_puck += puck.is_some() as usize;
        out.push(Position {
            frame: f.frame,
            time_s: f.time_s,
            homography: h.as_ref().map(round8),
            cut,
            view: view.as_str(),
            rejected,
            keypoints_used: inliers.iter().map(|&i| k.name(i).to_string()).collect(),
            players,
            puck,
        });
    }
    Ok((out, summary))
}

/// positions.json, positions.csv and homographies.csv in `out_dir`, as homography.py writes them.
pub fn write_outputs(positions: &[Position], out_dir: &Path) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(out_dir.join("positions.json"))?);
    serde_json::to_writer(&mut f, positions)?;
    f.flush()?;

    // Python's csv module: minimal quoting, "\r\n" line ends, floats as repr().
    let mut f = BufWriter::new(File::create(out_dir.join("positions.csv"))?);
    f.write_all(
        b"frame,time_s,object,track_id,class,jersey_number,player_name,confidence,x_ft,y_ft\r\n",
    )?;
    for p in positions {
        for pl in &p.players {
            let row = [
                p.frame.to_string(),
                py_float(p.time_s),
                "player".into(),
                pl.track_id.map(|t| t.to_string()).unwrap_or_default(),
                csv_field(&pl.cls),
                csv_field(pl.jersey_number.as_deref().unwrap_or("")),
                csv_field(pl.player_name.as_deref().unwrap_or("")),
                py_float(pl.confidence),
                py_float(pl.rink_xy[0]),
                py_float(pl.rink_xy[1]),
            ];
            writeln_crlf(&mut f, &row)?;
        }
        if let Some(puck) = &p.puck {
            let row = [
                p.frame.to_string(),
                py_float(p.time_s),
                "puck".into(),
                String::new(),
                "puck".into(),
                String::new(),
                String::new(),
                py_float(puck.confidence),
                py_float(puck.rink_xy[0]),
                py_float(puck.rink_xy[1]),
            ];
            writeln_crlf(&mut f, &row)?;
        }
    }
    f.flush()?;

    let mut f = BufWriter::new(File::create(out_dir.join("homographies.csv"))?);
    f.write_all(
        b"frame,time_s,view,rejected,keypoints_used,h00,h01,h02,h10,h11,h12,h20,h21,h22\r\n",
    )?;
    for p in positions {
        let mut row = vec![
            p.frame.to_string(),
            py_float(p.time_s),
            p.view.to_string(),
            p.rejected.unwrap_or("").to_string(),
            csv_field(&p.keypoints_used.join(" ")),
        ];
        match &p.homography {
            Some(h) => row.extend(h.iter().flatten().map(|v| py_float(*v))),
            None => row.extend(std::iter::repeat(String::new()).take(9)),
        }
        writeln_crlf(&mut f, &row)?;
    }
    f.flush()
}

fn writeln_crlf(f: &mut impl Write, row: &[String]) -> std::io::Result<()> {
    f.write_all(row.join(",").as_bytes())?;
    f.write_all(b"\r\n")
}
