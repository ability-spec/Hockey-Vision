//! Print what the port does on one frame: matches, line candidates and the fit.
//!     cargo run --release --example debug_frame -- <detections.json> <frame> <W>x<H>
use hv_core::{detections, homography as hg, rink::keypoints};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let frames = detections::load(a[1].as_ref()).unwrap();
    let idx: usize = a[2].parse().unwrap();
    let (w, h) = a[3].split_once('x').unwrap();
    let fs = (w.parse().unwrap(), h.parse().unwrap());
    let k = keypoints();
    let (m, l, v) = hg::match_keypoints(&frames[idx], fs);
    println!("view {:?}", v);
    for x in &m {
        println!(
            "match {} {:?} w={} {:?}",
            k.name(x.keypoint),
            x.image_xy,
            x.weight,
            x.source
        );
    }
    for x in &l {
        println!(
            "line far={} near={} box={:?}",
            k.name(x.far.0),
            k.name(x.near.0),
            x.bx
        );
    }
    println!(
        "first robust fit: {:?}",
        hg::robust_fit(&m, fs).map(|(h, inl, e)| (
            h,
            inl.iter()
                .map(|i| k.name(i.keypoint).to_string())
                .collect::<Vec<_>>(),
            e
        ))
    );
    let (hm, inl) = hg::compute_homography(&m, &l, fs);
    println!("H {:?}", hm);
    println!(
        "inliers {:?}",
        inl.iter().map(|i| k.name(i.keypoint)).collect::<Vec<_>>()
    );
}
