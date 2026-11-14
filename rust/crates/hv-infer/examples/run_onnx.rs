//! Run one ONNX model on a raw float32 input (n,3,h,w) and write the raw float32 output, for
//! comparing ONNX Runtime versions: run_onnx <model.onnx> <in.f32> <n> <h> <w> <out.f32>
use hv_infer::model::{Device, Model};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut m = Model::load(a[1].as_ref(), Device::Cpu, 4).unwrap();
    let data: Vec<f32> = std::fs::read(&a[2]).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (shape, out) = m.run(data, a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap()).unwrap();
    eprintln!("output shape {shape:?}");
    std::fs::write(&a[6], out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
