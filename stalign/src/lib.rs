//! CTC alignment of an EPUB against an audiobook.
//!
//! The acoustic model is the ONNX MMS-300M forced aligner (31 Latin letters).
//! CUDA is used when the ONNX Runtime CUDA provider loads; otherwise the same
//! binary runs the model on CPU. The Viterbi and the anchor fit follow the
//! algorithm in the MIT-licensed Storyteller aligner.

#[macro_export]
macro_rules! say {
    ($($arg:tt)*) => {{
        use std::io::Write;
        let mut err = std::io::stderr().lock();
        let _ = writeln!(err, $($arg)*);
        let _ = err.flush();
    }};
}

pub mod audio;
pub mod ctc;
pub mod emit;
pub mod epub;
pub mod exclude;
pub mod pipeline;
pub mod runtime;
pub mod xmltree;

pub use pipeline::{align_book, emit_audio, run, Granularity, RunOptions};
