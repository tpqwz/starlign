# Restore points

Bookmarks starting with `restore-` are immutable. `jj bookmark list` shows where they point. Change ids below stay put when a commit is rewritten; these commits are not rewritten, because the bookmarks are immutable.

| Bookmark | Change | Keeps |
| --- | --- | --- |
| `restore-2026-10-24-pre-settings-release` | `nrnsxnnn` | starlign source with the Settings view (size slider, Show Align and Emit). The launch binary at that moment was the 2026-10-24 01:47 build, which does not contain Settings. A byte-identical copy is `stalign/dist/starlign-2026-10-24-0147` (sha256 `b9790455a6354f871d4fbfcfb56a140ec8889bd1b14209552ee4fee6f3aa1935`). |
| `restore-2026-10-24-settings-in-dist` | `wozpmqnr` | The same source, plus this ledger. `stalign/dist/starlign` is the release built 2026-10-24 03:52 (sha256 `e1a4c0612429362b5575b1c560fdc0ec6b5262ebabc7f8f74880d0637199351d`, 17825888 bytes). It contains "Show Align and Emit". The binary itself is not in the snapshot. `starlign --version` prints `starlign 0.1.0`. |

The window started before the 03:52 replace keeps the 01:47 inode until it exits (`/proc/<pid>/exe` ends in `(deleted)`). The next start of `dist/starlign` is the Settings build. Books on disk are not part of either snapshot: `aligned.epub`, the source EPUB, the GUI aligned EPUB, and `/tmp/recovery-aligned.epub`.
