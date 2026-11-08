#!/bin/sh
# Builds js/dist/engine.wasm from this crate and the engine (../zero-talk), with a pinned toolchain and path remaps, so that the
# same sources give the same bytes on any machine.
#
#   build.sh            builds, copies the file to js/dist/engine.wasm and writes its size and sha256 into js/dist/model.json
#                       and js/dist/files.json (needs Node.js, which the npm package needs anyway)
#   build.sh --check    builds into a temporary folder and fails unless the bytes equal js/dist/engine.wasm and match its
#                       size and sha256 in model.json and files.json; changes nothing in the repository
#
# Needs the toolchain named below with the wasm32-unknown-unknown target:
#   rustup toolchain install 1.99.1 --profile minimal --target wasm32-unknown-unknown
set -eu

TOOLCHAIN=1.99.1

HERE="$(cd "$(dirname "$0")" && pwd)"
RUST="$(cd "$HERE/../.." && pwd)"
DIST="$(cd "$RUST/../js/dist" && pwd)"
MODE="${1:-build}"
case "$MODE" in build | --check) ;; *) echo "usage: build.sh [--check]" >&2; exit 2 ;; esac

if command -v rustup >/dev/null 2>&1 && rustup toolchain list | grep -q "^$TOOLCHAIN"; then
    export RUSTUP_TOOLCHAIN="$TOOLCHAIN"
fi
have="$(rustc --version | cut -d' ' -f2)"
if [ "$have" != "$TOOLCHAIN" ]; then
    echo "build.sh: rustc $have found, $TOOLCHAIN needed: rustup toolchain install $TOOLCHAIN --profile minimal --target wasm32-unknown-unknown" >&2
    exit 2
fi

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
size() { wc -c < "$1" | tr -d ' '; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Paths that reach the module (panic locations) read /src, /cargo and /build, not the builder's folders. The more specific mapping
# comes last, because rustc applies the later mapping first.
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
RUSTFLAGS="-C target-feature=+simd128 --remap-path-prefix $HOME=/build --remap-path-prefix $CARGO_HOME_DIR=/cargo --remap-path-prefix $RUST=/src" \
CARGO_TARGET_DIR="$WORK/target" \
    cargo build --release --locked --target wasm32-unknown-unknown --manifest-path "$RUST/Cargo.toml" -p zero-talk-wasm
BUILT="$WORK/target/wasm32-unknown-unknown/release/zero_talk_wasm.wasm"
bytes="$(size "$BUILT")"
sum="$(sha256 "$BUILT")"

if [ "$MODE" = "build" ]; then
    cp "$BUILT" "$DIST/engine.wasm"
    node - "$DIST" "$bytes" "$sum" <<'JS'
const fs = require('node:fs');
const [dist, bytes, sum] = [process.argv[2], Number(process.argv[3]), process.argv[4]];
const model = JSON.parse(fs.readFileSync(`${dist}/model.json`, 'utf8'));
const entry = model.files.find((f) => f.name === 'engine.wasm');
entry.bytes = bytes;
entry.sha256 = sum;
fs.writeFileSync(`${dist}/model.json`, JSON.stringify(model, null, 2) + '\n');
const files = JSON.parse(fs.readFileSync(`${dist}/files.json`, 'utf8'));
files.files['engine.wasm'] = bytes;
files.total = Object.values(files.files).reduce((a, b) => a + b, 0);
const body = Object.entries(files.files).map(([k, v]) => `"${k}": ${v}`).join(', ');
fs.writeFileSync(`${dist}/files.json`, `{"total": ${files.total}, "files": {${body}}}\n`);
JS
    echo "engine.wasm: $bytes bytes, sha256 $sum"
    exit 0
fi

fail=0
node - "$DIST" "$bytes" "$sum" <<'JS' || fail=1
const fs = require('node:fs');
const [dist, bytes, sum] = [process.argv[2], Number(process.argv[3]), process.argv[4]];
const model = JSON.parse(fs.readFileSync(`${dist}/model.json`, 'utf8')).files.find((f) => f.name === 'engine.wasm');
const sizes = JSON.parse(fs.readFileSync(`${dist}/files.json`, 'utf8')).files['engine.wasm'];
let bad = 0;
const want = (what, got, expected) => {
    if (got !== expected) { console.error(`engine.wasm ${what}: built ${got}, recorded ${expected}`); bad = 1; }
};
want('bytes in model.json', bytes, model.bytes);
want('sha256 in model.json', sum, model.sha256);
want('bytes in files.json', bytes, sizes);
process.exit(bad);
JS
if [ "$(sha256 "$DIST/engine.wasm")" != "$sum" ]; then
    echo "engine.wasm: js/dist/engine.wasm is not what the sources build (sha256 $sum)" >&2
    fail=1
fi
if [ "$fail" -ne 0 ]; then
    echo "run rust/crates/zero-talk-wasm/build.sh to rebuild it and record the new size and sha256" >&2
    exit 1
fi
echo "engine.wasm: $bytes bytes, sha256 $sum, as built from the sources"
