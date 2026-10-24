# Restoring this tree

`/home/yy/code/st` is a Jujutsu repository (`jj` 0.46), stored in a colocated Git repo (`.jj` and `.git`). Named restore points are bookmarks whose names start with `restore-`. Repo config marks those bookmarks immutable, so `jj` refuses to rewrite them.

Use `jj` for snapshots. Leave `.git` as storage. A `git commit` beside `jj` rewrites the same commits and is easy to tangle.

The ledger of points is [restore-points.md](restore-points.md).

## See the points

```bash
jj bookmark list
jj log
```

## Start new work from a restore point

This keeps the bookmark where it is and checks that snapshot out as the parent of a new working-copy change:

```bash
jj new restore-2026-10-24-pre-settings-release
```

Later commits stay in `jj log`. This does not delete them.

## Make the working copy match a restore point

```bash
jj restore --from restore-2026-10-24-pre-settings-release
```

That copies tracked file contents into the current change. It does not move the bookmark.

## Undo a jj command

```bash
jj op log
jj op restore <operation-id>
```

`jj op restore` puts the repo back to that operation, including bookmarks and the working copy. `jj restore` only copies file contents.

## What a restore does not touch

These are ignored. A restore leaves them on disk and cannot recreate them:

| Path | Why it is outside history |
| --- | --- |
| `models/mms-fp16.onnx` | MMS weights, CC-BY-NC 4.0 |
| `stalign/runtime/lib/` | ONNX Runtime and CUDA, about 1.5GB. `stalign/dist/lib` points here |
| `stalign/target/` | Cargo output |
| `stalign/dist/starlign` | Launch binary, about 17MB, over the 1MiB snapshot limit |
| `stalign/dist/starlign-2026-10-24-0147` | Copy of the 01:47 binary, sha256 `b9790455a6354f871d4fbfcfb56a140ec8889bd1b14209552ee4fee6f3aa1935` |
| `*.m4b`, `*.epub`, `emission/` | The audiobook, the source EPUB, both aligned EPUBs, and cached emissions |

Do not delete those to "clean up" a restore. The reference book is `aligned.epub`. The full-book alignment from emissions is `/tmp/recovery-aligned.epub`. The GUI test wrote `Recovery—The Sacred Art (The Art of Spiritual Living)-aligned.epub`.

To rebuild the launch binary from a checked-out source tree:

```bash
cd stalign
cargo build --offline --release
cp -a target/release/starlign dist/starlign
```

Copying onto `dist/starlign` while a window is open does not change that process. The running process keeps the inode it started with. The new binary is what the next start runs. `dist/lib` must still be the symlink to `../runtime/lib`.

## Make another restore point

On the change you want to keep:

```bash
jj describe -m "Restore point: what this snapshot is."
jj bookmark create restore-YYYY-MM-DD-short-name
jj new
```

`jj new` matters. The working copy is a commit, and further edits amend it. A bookmark on that commit would move with the amend until you start a child. After `jj new`, the bookmark stays on the parent. Immutable bookmarks cannot be amended anyway.

Then add a row to `docs/restore-points.md` on the new change.
