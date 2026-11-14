//! `detections.json` as written by `run_models.py`: one record per frame, each with every model's
//! detections. Unknown fields are ignored, so newer files still load.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Detection {
    #[serde(rename = "class")]
    pub cls: String,
    pub confidence: f64,
    pub box_xyxy: [f64; 4],
    #[serde(default)]
    pub track_id: Option<i64>,
    #[serde(default)]
    pub jersey_number: Option<String>,
    #[serde(default)]
    pub player_name: Option<String>,
    /// The player's team abbreviation (run_models.py --teams).
    #[serde(default)]
    pub team: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FrameRecord {
    pub frame: i64,
    pub time_s: f64,
    /// Camera cut, recorded by run_models.py while it has the frame decoded.
    #[serde(default)]
    pub cut: Option<bool>,
    /// run_models.py --rink-stride skipped the rink models on this frame.
    #[serde(default)]
    pub rink_skipped: bool,
    pub detections: HashMap<String, Vec<Detection>>,
}

impl FrameRecord {
    pub fn model(&self, name: &str) -> &[Detection] {
        self.detections.get(name).map(Vec::as_slice).unwrap_or(&[])
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DetectionsFile {
    Video(Vec<FrameRecord>),
    Image(HashMap<String, Vec<Detection>>),
}

/// Every frame of a detections.json. A single image's output (a plain dict) becomes one frame,
/// as in homography.py.
pub fn load(path: &Path) -> Result<Vec<FrameRecord>, String> {
    let mut text = String::new();
    BufReader::new(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .read_to_string(&mut text)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn parse(text: &str) -> Result<Vec<FrameRecord>, serde_json::Error> {
    Ok(match serde_json::from_str::<DetectionsFile>(text)? {
        DetectionsFile::Video(frames) => frames,
        DetectionsFile::Image(detections) => vec![FrameRecord {
            frame: 0,
            time_s: 0.0,
            cut: None,
            rink_skipped: false,
            detections,
        }],
    })
}
