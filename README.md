# starlign

This is beta. The tool is early, and it may still shift.

starlign lines up an audiobook with an EPUB and writes an EPUB 3 media overlay. The sound is put in as it was. It is not made again.

One Rust tool does the work. A window opens when you start it. The same tool also runs from the shell as `run`, `emit`, or `align`.

The window holds audiobook-and-EPUB pairs and runs them one after another. The book must already be an EPUB.

## Speech net

The net is MMS-300M, in ONNX, with 31 Latin letters. It was made for English books. The weights are not in this tree. They are CC-BY-NC 4.0. The tool fetches them only when you ask, and it says so. In the window, "Download the model if it is missing" starts on. From the shell, pass `--download`.

CUDA is used when that ONNX Runtime add-on loads from the folder beside the tool. Otherwise the same build runs on the CPU. A Mac build is CPU-only.

## What you get

An EPUB with the overlay and the sound inside. Times are by sentence, or by word if you ask. The window starts on sentences, and on Chapter split: one sound file for each lined-up chapter. Older read-aloud players follow the highlight more steadily that way. The split follows the EPUB, not chapter marks in the sound file. The shell keeps one sound file unless you pass `--split-chapters`, `--split-seconds`, or `--split-parts`.

## Build

From `starlign/`:

```bash
cargo build --release --locked
```

Linux cannot link a Mac build. See [docs/macos.md](docs/macos.md). The Mac workflow on GitHub runs only when someone starts it by hand.

This tree holds the source. The built tools are not kept here. They still need ONNX Runtime 1.22 beside them before they can run the net.

## Sharing

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The anchor fit, and the skip rules for spine items that are not read aloud, follow Storyteller's MIT aligner in `libraries/align`.

This page is in Anglish.
