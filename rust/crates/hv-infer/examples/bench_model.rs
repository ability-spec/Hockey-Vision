//! Time one ONNX model: bench_model <model.onnx> <n> <h> <w> <cpu|coreml> <threads> [runs]
use hv_infer::model::{Device, Model};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (n, h, w): (usize, usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    let dev = if a[5] == "coreml" { Device::CoreMl } else { Device::Cpu };
    let runs: usize = a.get(7).map_or(20, |v| v.parse().unwrap());
    let mut m = Model::load(a[1].as_ref(), dev, a[6].parse().unwrap()).unwrap();
    let data = vec![0.5f32; n * 3 * h * w];
    for _ in 0..3 { m.run(data.clone(), n, h, w).unwrap(); }
    let t = std::time::Instant::now();
    for _ in 0..runs { m.run(data.clone(), n, h, w).unwrap(); }
    println!("{:.2} ms per run ({} images of {}x{})", t.elapsed().as_secs_f64() * 1e3 / runs as f64, n, w, h);
}
