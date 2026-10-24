//! Locate ONNX Runtime and its CUDA libraries, then re-exec so the dynamic
//! linker sees them. `LD_LIBRARY_PATH` is read at process start, so it has to
//! be set before `libonnxruntime` is opened.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const READY: &str = "STARLIGN_LIBS_READY";

pub fn prepare() {
    if env::var_os(READY).is_some() {
        return;
    }
    let Some(dir) = runtime_lib_dir() else {
        return;
    };
    let ort = dir.join("libonnxruntime.so");
    if !ort.is_file() {
        return;
    }
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("starlign"));
    let joined = match env::var("LD_LIBRARY_PATH") {
        Ok(old) if !old.is_empty() => format!("{}:{}", dir.display(), old),
        _ => dir.display().to_string(),
    };
    let status = Command::new(exe)
        .args(env::args().skip(1))
        .env(READY, "1")
        .env("LD_LIBRARY_PATH", &joined)
        .env("ORT_DYLIB_PATH", &ort)
        .status();
    match status {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(err) => {
            crate::say!("failed to re-exec with ONNX Runtime libs: {err}");
        }
    }
}

pub fn runtime_lib_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    // STARLIGN_RUNTIME is the override. STALIGN_RUNTIME is the previous name.
    if let Some(dir) = env::var_os("STARLIGN_RUNTIME").or_else(|| env::var_os("STALIGN_RUNTIME")) {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("lib"));
            candidates.push(parent.join("runtime").join("lib"));
            // target/debug/starlign -> ../../runtime/lib when developing in-tree
            candidates.push(parent.join("..").join("..").join("runtime").join("lib"));
        }
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime/lib"));
    candidates.into_iter().find(|dir| dir.join("libonnxruntime.so").is_file())
}

pub fn using_runtime() -> Option<PathBuf> {
    runtime_lib_dir()
}

pub fn describe_runtime() -> String {
    match runtime_lib_dir() {
        Some(dir) => format!("{}", dir.display()),
        None => "system ONNX Runtime".to_string(),
    }
}

#[allow(dead_code)]
pub fn lib_dir_has_cuda(dir: &Path) -> bool {
    dir.join("libonnxruntime_providers_cuda.so").is_file()
}
