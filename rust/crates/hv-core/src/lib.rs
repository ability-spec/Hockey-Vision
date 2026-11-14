//! Hockey-Vision's per-frame CPU work in Rust.
//!
//! This is phase 1 of the port described in `rust/ANALYSIS.md`: the parts of the pipeline that
//! don't need the models. `homography` + `pipeline` replace `homography.py` (without its optional
//! videos) and read and write the same files; `jersey` holds run_models.py's jersey-number logic for
//! the Rust inference loop to come. `geometry` reproduces the OpenCV calls they need.

pub mod bytetrack;
pub mod columns;
pub mod detections;
pub mod events;
pub mod export;
pub mod geometry;
pub mod homography;
pub mod image;
pub mod jersey;
pub mod pipeline;
pub mod pyfmt;
pub mod pyval;
pub mod rink;
pub mod roster;
pub mod yolo;
