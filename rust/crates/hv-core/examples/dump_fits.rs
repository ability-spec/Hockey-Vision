//! Each frame's own fit before smoothing, one JSON line per frame (for parity debugging).
//!     cargo run --release --example dump_fits -- <detections.json> <W>x<H> > fits.jsonl
use hv_core::{detections, pipeline, rink::keypoints};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let frames = detections::load(a[1].as_ref()).unwrap();
    let (w, h) = a[2].split_once('x').unwrap();
    let fs = (w.parse().unwrap(), h.parse().unwrap());
    let k = keypoints();
    for f in &frames {
        let fit = pipeline::fit_frame(f, fs);
        let h = fit.h.map(|h| {
            (0..3)
                .map(|r| (0..3).map(|c| h[(r, c)]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        });
        let kps: Vec<&str> = fit.inliers.iter().map(|&i| k.name(i)).collect();
        println!(
            "{}",
            serde_json::json!({"frame": f.frame, "H": h, "inliers": kps})
        );
    }
}
