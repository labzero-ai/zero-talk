"""Reject build paths in release archives and binaries without printing matches."""

import argparse
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import zipfile


FORBIDDEN = (b"/home/runner", b"/Users/", b"runneradmin", b"C:\\Users", b"C:/Users")
BINARY_SUFFIXES = {".so", ".pyd", ".dll", ".dylib", ".exe", ".wasm", ".a", ".rlib"}


def check_bytes(name, data):
    # Also catch UTF-16 paths in Windows binaries; strings normally reports ASCII.
    folded = data.lower()
    if any(value.lower() in folded or value.decode().lower().encode("utf-16le") in folded
           for value in FORBIDDEN):
        raise SystemExit("Forbidden build path in artifact; fix path remapping and rebuild.")
    binary = Path(name).suffix.lower() in BINARY_SUFFIXES or data[:4] in (
        b"\x7fELF", b"\x00asm", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
        b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe",
    ) or data[:2] == b"MZ"
    if binary:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "binary"
            path.write_bytes(data)
            result = subprocess.run(["strings", "-a", str(path)], capture_output=True, check=True)
        if any(value.lower() in result.stdout.lower() for value in FORBIDDEN):
            raise SystemExit("strings found a forbidden build path; fix path remapping and rebuild.")


def check_archive(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                check_bytes(member.filename, member.filename.encode())
                if not member.is_dir():
                    check_bytes(member.filename, archive.read(member))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive:
                check_bytes(member.name, member.name.encode())
                check_bytes(member.name, member.linkname.encode())
                if member.isfile():
                    check_bytes(member.name, archive.extractfile(member).read())
    else:
        check_bytes(path.name, path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if not shutil.which("strings"):
        raise SystemExit("Install binutils (strings) before checking release artifacts.")
    for path in args.paths:
        if not path.is_file():
            raise SystemExit("A release artifact is missing; build it before checking.")
        check_archive(path)
    print(f"Build-path check passed for {len(args.paths)} artifact(s).")


if __name__ == "__main__":
    main()
