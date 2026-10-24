//! CTC audiobook alignment. One binary, ONNX Runtime, CUDA when the driver is there.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use starlign::emit::Device;
use starlign::pipeline::{self, Granularity, RunOptions};
use starlign::runtime;
use std::path::PathBuf;

#[cfg(feature = "gui")]
mod gui;

#[derive(Parser)]
#[command(
    name = "starlign",
    version,
    about = "Align an EPUB to an audiobook with MMS-300M CTC and write EPUB 3 Media Overlays.",
    long_about = "Align an EPUB to an audiobook and write EPUB 3 Media Overlays (SMIL) plus sentence or word spans.\n\nWith no command, a window opens for the same options. `starlign gui` opens it explicitly. The window queues audiobook and EPUB pairs and runs them in order.\n\nThe acoustic model is the ONNX MMS-300M forced aligner (CC-BY-NC 4.0, 31 Latin letters). CUDA is used when libonnxruntime_providers_cuda.so loads from the runtime directory; otherwise the same binary runs the model on CPU.\n\nQuartzNet and Citrinet are not included. Their published Storyteller weights are safetensors for a private loader, not ONNX."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Decode audio, emit CTC log-probabilities, align the EPUB, and write SMIL.
    Run(RunArgs),
    /// Decode audio and write an emissions file. Skips the EPUB.
    Emit(EmitArgs),
    /// Align an EPUB using an emissions file from `emit`.
    Align(AlignArgs),
    /// Open a window for choosing the audiobook, the EPUB, and the other options.
    Gui,
}

#[derive(Parser)]
struct RunArgs {
    #[arg(long)]
    audiobook: PathBuf,
    #[arg(long)]
    epub: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    /// Cache of frame log-probabilities. Reused when it already exists.
    #[arg(long)]
    emissions: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,
    #[arg(long, value_enum, default_value_t = GranularityArg::Sentence)]
    granularity: GranularityArg,
    #[arg(long)]
    model: Option<PathBuf>,
    /// Do not fold a silence longer than this into the previous span. 0 keeps every gap.
    #[arg(long, default_value_t = 3.0)]
    gap_seconds: f64,
    /// Leave spans with a dropped CTC path out of the SMIL.
    #[arg(long, default_value_t = false)]
    drop_unread: bool,
    /// Only the first N seconds of audio. Useful for a short check.
    #[arg(long)]
    max_seconds: Option<f64>,
    #[arg(long)]
    reports: Option<PathBuf>,
    /// Download MMS-300M fp16 ONNX if --model is missing. The weights are CC-BY-NC 4.0.
    #[arg(long, default_value_t = false)]
    download: bool,
}

#[derive(Parser)]
struct EmitArgs {
    #[arg(long)]
    audiobook: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    max_seconds: Option<f64>,
    #[arg(long, default_value_t = false)]
    download: bool,
}

#[derive(Parser)]
struct AlignArgs {
    #[arg(long)]
    audiobook: PathBuf,
    #[arg(long)]
    epub: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long)]
    emissions: PathBuf,
    #[arg(long, value_enum, default_value_t = GranularityArg::Sentence)]
    granularity: GranularityArg,
    #[arg(long, default_value_t = 3.0)]
    gap_seconds: f64,
    #[arg(long, default_value_t = false)]
    drop_unread: bool,
    #[arg(long)]
    max_seconds: Option<f64>,
    #[arg(long)]
    reports: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
}

impl DeviceArg {
    fn into_device(self) -> Device {
        match self {
            DeviceArg::Auto => Device::Auto,
            DeviceArg::Cpu => Device::Cpu,
            DeviceArg::Cuda => Device::Cuda,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum GranularityArg {
    Sentence,
    Word,
}

impl GranularityArg {
    fn into_granularity(self) -> Granularity {
        match self {
            GranularityArg::Sentence => Granularity::Sentence,
            GranularityArg::Word => Granularity::Word,
        }
    }
}

fn main() {
    runtime::prepare();
    if let Err(err) = dispatch() {
        starlign::say!("error: {err:#}");
        std::process::exit(1);
    }
}

fn dispatch() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None | Some(Command::Gui) => open_window(),
        Some(Command::Run(args)) => {
            let opts = RunOptions {
                audiobook: args.audiobook,
                epub: args.epub,
                output: args.output,
                emissions: args.emissions,
                device: args.device.into_device(),
                granularity: args.granularity.into_granularity(),
                model: args.model.unwrap_or_else(pipeline::default_model_path),
                gap_seconds: args.gap_seconds,
                drop_unread: args.drop_unread,
                max_seconds: args.max_seconds,
                reports: args.reports,
                download: args.download,
            };
            pipeline::run(&opts)
        }
        Some(Command::Emit(args)) => pipeline::emit_audio(
            &args.audiobook,
            &args.output,
            &args.model.unwrap_or_else(pipeline::default_model_path),
            args.device.into_device(),
            args.max_seconds,
            args.download,
        ),
        Some(Command::Align(args)) => {
            let opts = RunOptions {
                audiobook: args.audiobook,
                epub: args.epub,
                output: args.output,
                emissions: Some(args.emissions),
                device: Device::Cpu,
                granularity: args.granularity.into_granularity(),
                model: PathBuf::new(),
                gap_seconds: args.gap_seconds,
                drop_unread: args.drop_unread,
                max_seconds: args.max_seconds,
                reports: args.reports,
                download: false,
            };
            pipeline::align_book(&opts)
        }
    }
}

fn open_window() -> Result<()> {
    #[cfg(feature = "gui")]
    {
        gui::launch()
    }
    #[cfg(not(feature = "gui"))]
    {
        anyhow::bail!("this build has no window. Use `starlign run`, `starlign align`, or `starlign emit`.")
    }
}
