#!/bin/bash
# Pack starlign with the official ONNX Runtime 1.22 CPU library beside it.
# Usage: pack-release.sh <rust-target> <ort-archive-name> <tgz|zip>
set -euo pipefail

target="${1:?target}"
archive="${2:?archive}"
ext="${3:?ext}"

py() {
  if command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  else
    python "$@"
  fi
}

url="https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/${archive}.${ext}"
curl -fL --retry 3 -o ort.pkg "$url"
rm -rf ort-src
mkdir ort-src
if [ "$ext" = "zip" ]; then
  # Git's tar on Windows does not unpack zip files.
  py - ort.pkg ort-src << 'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PY
else
  tar -xf ort.pkg -C ort-src
fi

libdir=""
while IFS= read -r dir; do
  libdir="$dir"
  break
done < <(find ort-src -type d -name lib)
if [ -z "$libdir" ]; then
  echo "ONNX Runtime archive has no lib directory" >&2
  find ort-src | head
  exit 1
fi

stage="starlign-${target}"
rm -rf "$stage"
mkdir -p "$stage/lib"
found=0
for file in "$libdir"/*; do
  base="$(basename "$file")"
  case "$base" in
    *.so | *.so.* | *.dll | *.dylib)
      cp -L "$file" "$stage/lib/$base"
      found=1
      ;;
  esac
done
if [ "$found" -ne 1 ]; then
  echo "ONNX Runtime archive has no shared library" >&2
  find "$libdir" -maxdepth 1 | head
  exit 1
fi

bindir="starlign/target/${target}/release"
if [ -f "${bindir}/starlign.exe" ]; then
  cp -a "${bindir}/starlign.exe" "$stage/starlign.exe"
elif [ -f "${bindir}/starlign" ]; then
  cp -a "${bindir}/starlign" "$stage/starlign"
  chmod +x "$stage/starlign"
else
  echo "built binary not found in ${bindir}" >&2
  exit 1
fi

rm -f "${stage}.zip"
py - "$stage" << 'PY'
import os, sys, zipfile
stage = sys.argv[1]
out = stage + ".zip"
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for dirpath, _, names in os.walk(stage):
        for name in names:
            full = os.path.join(dirpath, name)
            arc = os.path.relpath(full, ".").replace("\\", "/")
            zf.write(full, arc)
PY
test -s "${stage}.zip"
