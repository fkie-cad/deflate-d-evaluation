#!/usr/bin/env bash
#
# Build a pinned GNU coreutils at the canonical -g -O2 and emit stripped
# binaries plus a provenance manifest for the token-savings study.
#
# Why -O2: GNU Autoconf sets CFLAGS="-g -O2" for GCC when the builder does not
# override it, so -O2 is coreutils' own default build. We therefore do NOT pass
# CFLAGS and let the autotools default stand (recorded as opt_level "O2" in the
# manifest).
#
# Output: <out-dir>/<prog> binaries + <out-dir>/corpus_manifest.json
#
#   evaluation/corpora/coreutils.sh [out-dir]
#   COREUTILS_VERSION=9.5 evaluation/corpora/coreutils.sh ./coreutils_bin
#
# NOTE on ISA: binaries are native to the build host (Mach-O arm64 on macOS,
# ELF x86-64 on a Linux box/container). For the paper's canonical ELF x86-64
# target, run this inside a Linux x86-64 environment (e.g. a Docker container).
set -euo pipefail

VERSION="${COREUTILS_VERSION:-9.5}"
OUT_DIR="${1:-$(pwd)/coreutils_bin}"
TARBALL="coreutils-${VERSION}.tar.xz"
URL="https://ftp.gnu.org/gnu/coreutils/${TARBALL}"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

WORK="$(mktemp -d /tmp/coreutils_build.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT_DIR"

echo "[coreutils] fetching $URL"
curl -fsSL "$URL" -o "$WORK/$TARBALL"
echo "[coreutils] extracting"
tar -C "$WORK" -xf "$WORK/$TARBALL"
SRC="$WORK/coreutils-${VERSION}"

echo "[coreutils] configure + make (autotools default CFLAGS = -g -O2)"
( cd "$SRC" && ./configure >/dev/null && make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" >/dev/null )

# Collect built program binaries (executables in src/, excluding libtool
# wrappers, scripts, and object/archive files).
echo "[coreutils] collecting + stripping binaries"
manifest_entries=""
count=0
for f in "$SRC"/src/*; do
    [ -f "$f" ] || continue
    [ -x "$f" ] || continue
    case "$f" in *.o|*.a|*.la|*.sh) continue;; esac
    # Skip text/script files; keep only real Mach-O/ELF executables.
    if ! file -b "$f" | grep -qiE "executable|Mach-O|ELF"; then continue; fi
    name="$(basename "$f")"
    cp "$f" "$OUT_DIR/$name"
    strip "$OUT_DIR/$name" 2>/dev/null || true
    sum="$(sha256 "$OUT_DIR/$name")"
    manifest_entries="${manifest_entries}    \"${name}\": {\"sha256\": \"${sum}\"},\n"
    count=$((count + 1))
done

# Trim trailing comma from the last JSON entry.
entries="$(printf "%b" "$manifest_entries" | sed '$ s/,$//')"
arch="$(uname -m)"
host="$(uname -s)"
tarball_sum="$(sha256 "$WORK/$TARBALL")"

cat > "$OUT_DIR/corpus_manifest.json" <<JSON
{
  "corpus": "coreutils",
  "version": "${VERSION}",
  "opt_level": "O2",
  "stripped": true,
  "arch": "${arch}",
  "host": "${host}",
  "source": "${URL}",
  "source_sha256": "${tarball_sum}",
  "binaries": {
${entries}
  }
}
JSON

echo "[coreutils] wrote $count binaries + manifest to $OUT_DIR"
