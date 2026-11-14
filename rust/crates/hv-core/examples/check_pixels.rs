//! Pixel parity helper for rust/bench/check_pixels.py: reads yuv420p frames and crop specs, writes
//! what hv_core::image makes of them (BGR frames, 640x360 downscales, resized crops, histograms).
//!
//!     check_pixels <frames.yuv> <w> <h> <n> <crops.txt> <out dir>
use std::io::Write;

use hv_core::image::{resize_linear, CutDetector, Image, YuvToBgr};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (w, h, n): (usize, usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    let yuv = std::fs::read(&a[1]).unwrap();
    let crops: Vec<Vec<usize>> = std::fs::read_to_string(&a[5])
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect();
    let out = std::path::Path::new(&a[6]);
    let conv = YuvToBgr::bt601_limited();
    let cuts = CutDetector::default();
    let (mut bgr, mut small, mut crop_out, mut hists) = (vec![], vec![], vec![], vec![]);
    let mut frames = vec![];
    for k in 0..n {
        let mut img = Image::default();
        conv.convert(&yuv[k * w * h * 3 / 2..(k + 1) * w * h * 3 / 2], w, h, &mut img);
        bgr.extend_from_slice(&img.data);
        small.extend_from_slice(&resize_linear(&img, 640, 360).data);
        for v in cuts.histogram(&img) {
            hists.extend_from_slice(&v.to_le_bytes());
        }
        frames.push(img);
    }
    for c in &crops {
        let (f, x1, y1, x2, y2, dw, dh) = (c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
        crop_out.extend_from_slice(&resize_linear(&frames[f].crop(x1, y1, x2, y2), dw, dh).data);
    }
    for (name, data) in [("bgr.raw", &bgr), ("small.raw", &small), ("crops.raw", &crop_out), ("hists.raw", &hists)] {
        std::fs::File::create(out.join(name)).unwrap().write_all(data).unwrap();
    }
}
