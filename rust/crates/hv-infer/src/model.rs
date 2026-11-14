//! A YOLO model exported to ONNX (`yolo export format=onnx dynamic=True`), run with ONNX Runtime.
//!
//! On the CPU one session takes any input shape. With CoreML (Apple GPU / Neural Engine) a model
//! is much faster with fixed shapes (about 4x for these models), so each shape the pipeline uses
//! all the time (`allow_static`) gets its own CoreML session with the dynamic dimensions pinned;
//! any other shape runs on the CPU session.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;

/// Which hardware runs the models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// Apple's CoreML (GPU / Neural Engine); build with --features coreml.
    CoreMl,
}

pub struct Model {
    #[allow(dead_code)]
    path: PathBuf,
    device: Device,
    #[allow(dead_code)]
    threads: usize,
    cpu: Session,
    #[allow(dead_code)]
    static_sessions: HashMap<(usize, usize, usize), Session>,
    static_shapes: Vec<(usize, usize, usize)>,
    pub names: Vec<String>,
    pub stride: usize,
    input: String,
}

/// "{0: 'goalie_a', 1: 'goalie_b'}" (the Python dict Ultralytics writes) -> class names by index.
fn parse_names(s: &str) -> Vec<String> {
    let mut out: Vec<(usize, String)> = vec![];
    for part in s.trim().trim_start_matches('{').trim_end_matches('}').split(", ") {
        let Some((k, v)) = part.split_once(": ") else { continue };
        let v = v.trim();
        let v = v.strip_prefix(['\'', '"']).and_then(|v| v.strip_suffix(['\'', '"'])).unwrap_or(v);
        if let Ok(k) = k.trim().parse() {
            out.push((k, v.to_string()));
        }
    }
    out.sort();
    out.into_iter().map(|(_, v)| v).collect()
}

fn builder(threads: usize) -> ort::Result<SessionBuilder> {
    Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?.with_intra_threads(threads)
}

impl Model {
    pub fn load(path: &Path, device: Device, threads: usize) -> Result<Self, String> {
        let err = |e: ort::Error| format!("{}: {e}", path.display());
        #[cfg(not(feature = "coreml"))]
        if device == Device::CoreMl {
            return Err("this build has no CoreML support; rebuild with --features coreml".into());
        }
        let cpu = builder(threads).and_then(|b| b.commit_from_file(path)).map_err(err)?;
        let (names, stride) = {
            let meta = cpu.metadata().map_err(err)?;
            let names = meta.custom("names").map_err(err)?.map(|s| parse_names(&s)).unwrap_or_default();
            let stride = meta.custom("stride").map_err(err)?.and_then(|s| s.trim().parse().ok()).unwrap_or(32);
            (names, stride)
        };
        if names.is_empty() {
            return Err(format!("{}: no class names in the model's metadata (export it with Ultralytics)", path.display()));
        }
        let input = cpu.inputs[0].name.clone();
        Ok(Model {
            path: path.to_path_buf(),
            device,
            threads,
            cpu,
            static_sessions: HashMap::new(),
            static_shapes: vec![],
            names,
            stride,
            input,
        })
    }

    /// With CoreML, run inputs of (n, h, w) on a fixed-shape CoreML session.
    pub fn allow_static(&mut self, n: usize, h: usize, w: usize) {
        if self.device == Device::CoreMl && !self.static_shapes.contains(&(n, h, w)) {
            self.static_shapes.push((n, h, w));
        }
    }

    pub fn is_static(&self, n: usize, h: usize, w: usize) -> bool {
        self.static_shapes.contains(&(n, h, w))
    }

    #[cfg(feature = "coreml")]
    fn static_session(&mut self, key: (usize, usize, usize)) -> Result<&mut Session, String> {
        use ort::execution_providers::coreml::{CoreMLComputeUnits, CoreMLModelFormat};
        use ort::execution_providers::CoreMLExecutionProvider;
        if !self.static_sessions.contains_key(&key) {
            let units = match std::env::var("HV_COREML_UNITS").as_deref() {
                Ok("gpu") => CoreMLComputeUnits::CPUAndGPU,
                Ok("ane") => CoreMLComputeUnits::CPUAndNeuralEngine,
                Ok("cpu") => CoreMLComputeUnits::CPUOnly,
                _ => CoreMLComputeUnits::All,
            };
            let ep = CoreMLExecutionProvider::default()
                .with_model_format(CoreMLModelFormat::MLProgram)
                .with_compute_units(units);
            let (n, h, w) = key;
            let s = builder(self.threads)
                .and_then(|b| b.with_dimension_override("batch", n as i64))
                .and_then(|b| b.with_dimension_override("height", h as i64))
                .and_then(|b| b.with_dimension_override("width", w as i64))
                .and_then(|b| b.with_execution_providers([ep.build().error_on_failure()]))
                .and_then(|b| b.commit_from_file(&self.path))
                .map_err(|e| format!("{}: {e}", self.path.display()))?;
            self.static_sessions.insert(key, s);
        }
        Ok(self.static_sessions.get_mut(&key).unwrap())
    }

    #[cfg(not(feature = "coreml"))]
    fn static_session(&mut self, _key: (usize, usize, usize)) -> Result<&mut Session, String> {
        Err("no CoreML support in this build".into())
    }

    /// Run a batch of `n` images of (h, w), CHW float32 back to back. Returns the raw output and
    /// its shape ([n, 4 + classes, anchors]).
    pub fn run(&mut self, data: Vec<f32>, n: usize, h: usize, w: usize) -> Result<(Vec<usize>, Vec<f32>), String> {
        let input_name = self.input.clone();
        let session = if self.is_static(n, h, w) { self.static_session((n, h, w))? } else { &mut self.cpu };
        let input = Tensor::from_array(([n, 3, h, w], data.into_boxed_slice())).map_err(|e| e.to_string())?;
        let outputs = session.run(ort::inputs![input_name.as_str() => input]).map_err(|e| e.to_string())?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        Ok((shape.iter().map(|&d| d as usize).collect(), values.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn names_from_metadata() {
        assert_eq!(super::parse_names("{0: 'goalie_a', 1: 'b', 10: 'x'}"), vec!["goalie_a", "b", "x"]);
    }
}
