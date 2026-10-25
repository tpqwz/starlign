//! Locate ONNX Runtime and its CUDA libraries, then re-exec so the dynamic
//! linker sees them. The library search path is read at process start, so it
//! has to be set before ONNX Runtime is opened.
//!
//! Linux and FreeBSD use `libonnxruntime.so` and `LD_LIBRARY_PATH`. macOS uses
//! `libonnxruntime.dylib` and `DYLD_LIBRARY_PATH`. Windows uses
//! `onnxruntime.dll` and `PATH`.

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
    let ort = dir.join(ort_library_file());
    if !ort.is_file() {
        return;
    }
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("starlign"));
    let joined = prepend_search_path(&dir, env::var(library_search_var()).ok().as_deref());
    let status = Command::new(exe)
        .args(env::args().skip(1))
        .env(READY, "1")
        .env(library_search_var(), &joined)
        .env("ORT_DYLIB_PATH", &ort)
        .status();
    match status {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(err) => {
            crate::say!("failed to re-exec with ONNX Runtime libs: {err}");
        }
    }
}

pub fn ort_library_file() -> String {
    dylib("onnxruntime")
}

pub fn cuda_provider_file() -> String {
    dylib("onnxruntime_providers_cuda")
}

pub fn shared_provider_file() -> String {
    dylib("onnxruntime_providers_shared")
}

/// Environment variable the dynamic linker reads for this target.
pub fn library_search_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

pub fn prepend_search_path(dir: &Path, old: Option<&str>) -> String {
    let dir = dir.display().to_string();
    match old {
        Some(old) if !old.is_empty() => format!("{dir}{}{old}", search_path_sep()),
        _ => dir,
    }
}

fn search_path_sep() -> char {
    if cfg!(windows) {
        ';'
    } else {
        ':'
    }
}

fn dylib(name: &str) -> String {
    format!(
        "{}{name}{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_SUFFIX
    )
}

pub fn runtime_lib_dir() -> Option<PathBuf> {
    let library = ort_library_file();
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
    candidates
        .into_iter()
        .find(|dir| dir.join(&library).is_file())
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
    dir.join(cuda_provider_file()).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn unix_runtime_names_and_search_path() {
        assert_eq!(ort_library_file(), "libonnxruntime.so");
        assert_eq!(cuda_provider_file(), "libonnxruntime_providers_cuda.so");
        assert_eq!(shared_provider_file(), "libonnxruntime_providers_shared.so");
        assert_eq!(library_search_var(), "LD_LIBRARY_PATH");
        assert_eq!(search_path_sep(), ':');
    }

    #[test]
    fn prepends_library_dir() {
        let dir = Path::new("/opt/starlign/lib");
        assert_eq!(prepend_search_path(dir, None), "/opt/starlign/lib");
        assert_eq!(prepend_search_path(dir, Some("")), "/opt/starlign/lib");
        assert_eq!(
            prepend_search_path(dir, Some("/usr/lib")),
            "/opt/starlign/lib:/usr/lib"
        );
    }
}
