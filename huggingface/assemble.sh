#!/bin/sh
# Usage: sh huggingface/assemble.sh OUTPUT_DIRECTORY
# Requires Python 3. Copies files locally; uploads nothing.
set -eu

if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo 'Usage: sh huggingface/assemble.sh OUTPUT_DIRECTORY' >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo 'Python 3 is required. Install it and run this script again.' >&2
    exit 1
fi
root=$(CDPATH= cd -P "$(dirname "$0")/.." && pwd)
exec python3 - "$root" "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys


def fail(message):
    sys.exit("assemble: " + message)


root, output = map(Path, sys.argv[1:])
# Keep the published bytes, with conventional licence names on Hugging Face.
names = {
    "model.bin": "model.bin",
    "frontend.bin": "frontend.bin",
    "LICENSE.txt": "LICENSE",
    "NOTICE.txt": "NOTICE",
    "THIRD-PARTY-NOTICES.txt": "THIRD-PARTY-NOTICES",
}
try:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        fail("Output must be a new or empty directory, not a file or symlink.")
    if output.exists() and any(output.iterdir()):
        fail("Output directory is not empty. Choose a new or empty directory.")
    dist = root / "js" / "dist"
    manifest_bytes = (dist / "model.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["id"] != "labzero-ai/zero-talk-11m":
        fail("Wrong model ID. Use the Zero Talk 11M release files.")
    entries = {entry["name"]: entry for entry in manifest["files"]}

    def verify(path, entry):
        if (path.stat().st_size != entry["bytes"] or
                hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]):
            fail("Size or SHA-256 mismatch for " + path.name +
                 ". Restore matching release files and run again.")

    # Check all upload payloads before creating the output directory.
    for source in names:
        verify(dist / source, entries[source])
    card = root / "huggingface" / "README.md"
    if not card.is_file():
        fail("Missing huggingface/README.md. Restore the model card and run again.")
    output.mkdir(parents=True, exist_ok=True)
    for source, destination in names.items():
        shutil.copyfile(dist / source, output / destination)
        verify(output / destination, entries[source])
    # The manifest cannot hash itself; the card is not a manifest entry.
    for source, destination in ((dist / "model.json", "model.json"), (card, "README.md")):
        shutil.copyfile(source, output / destination)
        if source.read_bytes() != (output / destination).read_bytes():
            fail("Copy mismatch for " + destination + ". Choose a fresh output directory and retry.")
    if (output / "model.json").read_bytes() != manifest_bytes:
        fail("Release manifest changed during assembly. Choose a fresh output directory and retry.")
    for path in sorted(output.iterdir()):
        print(path.name + "\t" + str(path.stat().st_size) + " bytes")
    print("Verified 5 manifest SHA-256 values; model.json and README.md copied byte for byte.")
except (OSError, ValueError, KeyError, TypeError):
    fail("Cannot assemble files. Check the release manifest, required files and directory permissions.")
PY
