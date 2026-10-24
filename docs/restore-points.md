# Restore points

Bookmarks starting with `restore-` are immutable. `jj bookmark list` shows where they point. Change ids below stay put when a commit is rewritten; these commits are not rewritten, because the bookmarks are immutable.

| Bookmark | Change | Keeps |
| --- | --- | --- |
| `restore-2026-10-24-pre-settings-release` | `nrnsxnnn` | starlign source with the Settings view (size slider, Show Align and Emit). `dist/starlign` at this moment is still the 2026-10-24 01:47 build, which does not contain Settings. That file is not in the snapshot. A byte-identical copy is `stalign/dist/starlign-2026-10-24-0147` (sha256 `b9790455a6354f871d4fbfcfb56a140ec8889bd1b14209552ee4fee6f3aa1935`). |

The window started from `dist/starlign` keeps running that 01:47 inode until it exits. Books on disk are not part of either snapshot: `aligned.epub`, the source EPUB, the GUI aligned EPUB, and `/tmp/recovery-aligned.epub`.
