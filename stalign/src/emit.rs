//! Run MMS-300M and stitch frame log-probabilities.
//!
//! Each window is 20 seconds of audio plus 2 seconds of context on each side.
//! Only frames whose receptive-field center falls inside the core are kept, and
//! each frame stores that center time so chunks do not drift apart.

use crate::audio::{read_pcm_range, SAMPLE_RATE};
use crate::ctc::{self, Emissions};
use crate::runtime;
use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::process::Command;
use std::os::unix::process::CommandExt;

const CORE_SECONDS: f64 = 20.0;
const CONTEXT_SECONDS: f64 = 2.0;
/// Center of MMS frame 0, in samples, for the unpadded conv stack
/// kernels [10,3,3,3,3,2,2] and strides [5,2,2,2,2,2,2]. Later frames step by 320.
const FRAME_CENTER0: f64 = 199.5;
const FRAME_HOP: f64 = 320.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cpu,
    Cuda,
}

pub fn emit_pcm(pcm: &Path, samples: u64, model_path: &Path, device: Device) -> Result<(Emissions, String)> {
    if samples == 0 {
        bail!("no audio to emit");
    }
    if !model_path.is_file() {
        bail!("model not found: {}", model_path.display());
    }
    let mut model = Model::open(model_path, device)?;
    let core = (CORE_SECONDS * SAMPLE_RATE as f64) as u64;
    let context = (CONTEXT_SECONDS * SAMPLE_RATE as f64) as u64;
    let mut data = Vec::new();
    let mut times = Vec::new();
    let mut classes = ctc::CLASSES;
    let mut core_start = 0u64;
    let started = std::time::Instant::now();
    while core_start < samples {
        let core_end = (core_start + core).min(samples);
        let win_start = core_start.saturating_sub(context);
        let win_end = (core_end + context).min(samples);
        let mut window = read_pcm_range(pcm, win_start, win_end)?;
        normalize(&mut window);
        let logits = match model.run(&window) {
            Ok(logits) => logits,
            Err(err) if model.device == "cuda" && device == Device::Auto => {
                // A second session in this process aborts inside ONNX Runtime
                // after the CUDA provider has already been created. Start a
                // fresh process on CPU instead.
                crate::say!("CUDA inference failed: {err}");
                crate::say!("relaunching on CPU");
                relaunch_on_cpu();
            }
            Err(err) => return Err(err),
        };
        if logits.classes != 0 {
            classes = logits.classes;
        }
        for frame in 0..logits.frames {
            let center = win_start as f64 + logits.center0 + frame as f64 * logits.hop;
            if center < core_start as f64 || center >= core_end as f64 {
                continue;
            }
            let row = &logits.data[frame * logits.classes..(frame + 1) * logits.classes];
            data.extend_from_slice(row);
            times.push(center / SAMPLE_RATE as f64);
        }
        let done = core_end as f64 / SAMPLE_RATE as f64;
        let total = samples as f64 / SAMPLE_RATE as f64;
        let rate = done / started.elapsed().as_secs_f64().max(0.001);
        crate::say!(
            "emit {done:.0}/{total:.0}s  {:.2}x  {}",
            rate,
            model.device
        );
        core_start = core_end;
    }
    if times.is_empty() {
        bail!("the acoustic model produced no frames");
    }
    ctc::log_softmax_rows(&mut data, classes);
    let device_name = model.device.clone();
    Ok((
        Emissions {
            classes,
            data,
            times,
        },
        device_name,
    ))
}

struct Logits {
    data: Vec<f32>,
    frames: usize,
    classes: usize,
    center0: f64,
    hop: f64,
}

struct Model {
    #[allow(dead_code)]
    env: ort::environment::Environment,
    session: Session,
    input_name: String,
    mask_name: Option<String>,
    input_f16: bool,
    mask_i64: bool,
    device: String,
}

impl Model {
    fn open(path: &Path, device: Device) -> Result<Self> {
        let env = environment(device != Device::Cpu)?;
        let mut opened = match device {
            Device::Cpu => None,
            Device::Cuda | Device::Auto => match open_session(&env, path, true) {
                Ok(session) => Some(session),
                Err(err) if device == Device::Auto => {
                    crate::say!("CUDA execution provider unavailable ({err:#}); using CPU");
                    None
                }
                Err(err) => return Err(err),
            },
        };
        let using_cuda = opened.is_some();
        if opened.is_none() {
            opened = Some(open_session(&env, path, false)?);
        }
        let session = opened.unwrap();
        let mut input_name = String::new();
        let mut mask_name = None;
        let mut input_f16 = false;
        let mut mask_i64 = true;
        crate::say!("model {}", path.display());
        for input in session.inputs() {
            let name = input.name().to_string();
            let dtype = format!("{:?}", input.dtype()).to_ascii_lowercase();
            crate::say!("  input {name} {dtype}");
            if name.to_ascii_lowercase().contains("mask") {
                mask_name = Some(name);
                mask_i64 = !dtype.contains("int32") || dtype.contains("int64");
            } else if input_name.is_empty() {
                input_name = name;
                input_f16 = dtype.contains("float16") || dtype.contains("f16");
            }
        }
        for output in session.outputs() {
            crate::say!("  output {} {:?}", output.name(), output.dtype());
        }
        if input_name.is_empty() {
            bail!("model has no inputs");
        }
        let device_name = if using_cuda { "cuda" } else { "cpu" };
        crate::say!("running on {device_name}");
        Ok(Self {
            env,
            session,
            input_name,
            mask_name,
            input_f16,
            mask_i64,
            device: device_name.to_string(),
        })
    }



    fn run(&mut self, samples: &[f32]) -> Result<Logits> {
        let frames_hint = conv_out_len(samples.len());
        let (center0, hop) = if frames_hint > 0 {
            (FRAME_CENTER0, FRAME_HOP)
        } else {
            (0.5, samples.len() as f64 / 1.0)
        };
        let outputs = self.run_session(samples)?;
        let output = &outputs[0];
        let (shape, data) = match output.try_extract_tensor::<f32>() {
            Ok((shape, data)) => {
                let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                (shape, data.to_vec())
            }
            Err(_) => {
                let (shape, data) = output.try_extract_tensor::<f16>().context("model output is neither f32 nor f16")?;
                let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data: Vec<f32> = data.iter().copied().map(f16::to_f32).collect();
                (shape, data)
            }
        };
        let (frames, classes) = match shape.as_slice() {
            [_, frames, classes] => (*frames, *classes),
            [frames, classes] => (*frames, *classes),
            other => bail!("unexpected logits shape {other:?}"),
        };
        if frames * classes != data.len() {
            bail!("logits length {} does not match {frames}x{classes}", data.len());
        }
        let hop = if frames_hint > 0 && frames > 0 {
            hop
        } else if frames > 0 {
            samples.len() as f64 / frames as f64
        } else {
            hop
        };
        let center0 = if frames_hint == frames { center0 } else { hop * 0.5 };
        Ok(Logits {
            data,
            frames,
            classes,
            center0,
            hop,
        })
    }

    fn run_session(&mut self, samples: &[f32]) -> Result<ort::session::SessionOutputs<'_>> {
        let input_name = self.input_name.clone();
        let mask_name = self.mask_name.clone();
        if self.input_f16 {
            let data: Vec<f16> = samples.iter().copied().map(f16::from_f32).collect();
            let values = Tensor::from_array(Array2::from_shape_vec((1, data.len()), data)?)?;
            return self.bind_and_run(input_name, values, mask_name, samples.len());
        }
        let values = Tensor::from_array(Array2::from_shape_vec((1, samples.len()), samples.to_vec())?)?;
        self.bind_and_run(input_name, values, mask_name, samples.len())
    }

    fn bind_and_run<T: ort::value::PrimitiveTensorElementType + std::fmt::Debug>(
        &mut self,
        input_name: String,
        values: Tensor<T>,
        mask_name: Option<String>,
        length: usize,
    ) -> Result<ort::session::SessionOutputs<'_>> {
        let outputs = if let Some(mask_name) = mask_name {
            if self.mask_i64 {
                let mask = Tensor::from_array(Array2::<i64>::ones((1, length)))?;
                self.session.run(ort::inputs![
                    input_name.as_str() => values,
                    mask_name.as_str() => mask,
                ])?
            } else {
                let mask = Tensor::from_array(Array2::<i32>::ones((1, length)))?;
                self.session.run(ort::inputs![
                    input_name.as_str() => values,
                    mask_name.as_str() => mask,
                ])?
            }
        } else {
            self.session.run(ort::inputs![input_name.as_str() => values])?
        };
        Ok(outputs)
    }
}

fn environment(preload_cuda: bool) -> Result<ort::environment::Environment> {
    let builder = if let Some(dir) = runtime::runtime_lib_dir() {
        if preload_cuda && runtime::lib_dir_has_cuda(&dir) {
            let _ = ort::ep::cuda::preload_dylibs(None, Some(dir.as_path()));
            let _ = ort::util::preload_dylib(dir.join("libonnxruntime_providers_shared.so"));
            let _ = ort::util::preload_dylib(dir.join("libonnxruntime_providers_cuda.so"));
        }
        ort::init_from(dir.join("libonnxruntime.so")).context("loading ONNX Runtime")?
    } else {
        ort::init()
    };
    builder
        .with_name("starlign")
        .with_telemetry(false)
        .build()
        .map_err(|err| anyhow!("onnx runtime: {err}"))
}

fn open_session(env: &ort::environment::Environment, path: &Path, cuda: bool) -> Result<Session> {
    // SessionBuilder errors carry the builder and are not Send, so they cannot
    // use `?` into anyhow. Convert them to a string here.
    let mut builder = Session::builder(env).map_err(|err| anyhow!("session: {err}"))?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    if cuda {
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default()
                .with_device_id(0)
                // Heuristic search fails on this cuDNN for the wav2vec positional
                // convolution (HEURISTIC_QUERY_FAILED). The default algorithm runs.
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Default)
                .build()])
            .map_err(|err| anyhow!("cuda provider: {err}"))?;
    } else {
        builder = builder
            .with_intra_threads(threads.min(16))
            .map_err(|err| anyhow!("cpu threads: {err}"))?;
    }
    builder
        .commit_from_file(path)
        .map_err(|err| anyhow!("loading {}: {err}", path.display()))
}

fn relaunch_on_cpu() -> ! {
    let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("starlign"));
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut replaced = false;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--device" && index + 1 < args.len() {
            args[index + 1] = "cpu".to_string();
            replaced = true;
            break;
        }
        if args[index].starts_with("--device=") {
            args[index] = "--device=cpu".to_string();
            replaced = true;
            break;
        }
        index += 1;
    }
    if !replaced {
        args.push("--device".to_string());
        args.push("cpu".to_string());
    }
    let err = Command::new(exe).args(&args).exec();
    crate::say!("failed to relaunch on CPU: {err}");
    std::process::exit(1);
}

fn normalize(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let n = samples.len() as f32;
    let mean = samples.iter().sum::<f32>() / n;
    let var = samples.iter().map(|sample| {
        let delta = *sample - mean;
        delta * delta
    }).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-7).sqrt();
    for sample in samples {
        *sample = (*sample - mean) * inv;
    }
}

fn conv_out_len(mut n: usize) -> usize {
    const KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
    const STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
    for index in 0..7 {
        if n < KERNELS[index] {
            return 0;
        }
        n = (n - KERNELS[index]) / STRIDES[index] + 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_length_matches_the_320_sample_hop() {
        let frames = conv_out_len(16_000);
        assert!(frames > 40 && frames < 60, "{frames}");
        let last = FRAME_CENTER0 + (frames - 1) as f64 * FRAME_HOP;
        assert!(last < 16_000.0, "{last}");
    }
}
