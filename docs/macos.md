# Building starlign for macOS

A Linux machine cannot link this program for a Mac. The Apple SDK stays on a Mac. Nothing in this tree downloads one.

The source already typechecks for `aarch64-apple-darwin` and `x86_64-apple-darwin`. The Mac build is CPU-only. There is no NVIDIA CUDA there. The window and the command line are the same binary.

## On a Mac

Install the Xcode Command Line Tools and Rust. From `starlign/`:

```bash
cargo build --release --locked
```

That binary matches the Mac it was built on. The other architecture:

```bash
rustup target add x86_64-apple-darwin
cargo build --release --locked --target x86_64-apple-darwin
```

On an Intel Mac, add `aarch64-apple-darwin` instead.

The program loads ONNX Runtime 1.22 by name. Download the official archive that matches the binary:

- Apple Silicon: <https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-arm64-1.22.0.tgz>
- Intel: <https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-x86_64-1.22.0.tgz>

Each archive contains `lib/libonnxruntime.dylib`. Put that `lib` directory next to the `starlign` binary:

```text
starlign
lib/libonnxruntime.dylib
lib/libonnxruntime.1.22.0.dylib
```

`STARLIGN_RUNTIME` can point at another directory that holds those dylibs. Do not commit the dylibs. The MMS weights are a separate CC-BY-NC 4.0 download and stay out of the repository too. The window can download them when "Download the model if it is missing" is checked.

## On GitHub

`.github/workflows/release.yml` builds Linux, Windows, and both Mac kinds on GitHub's machines. It runs on every push to `main`, and when someone starts it from the Actions tab. A tag named `v*` publishes that tag instead of refreshing `alpha`.

The Mac halves run on GitHub's Mac machines, which have the Apple SDK. Each release zip has the matching official ONNX Runtime 1.22 library in `lib/` next to the binary. The weights are not in the zip. CUDA is not in the zip. The release is marked as a prerelease.
