#!/usr/bin/env python3
"""Publish only the archives reviewed in the release run."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import urlopen


CRATES = ("zero-talk-model-11m", "zero-talk")
VERSION = os.environ["GITHUB_REF_NAME"].removeprefix("v")


def reviewed_digest(name):
    archive = Path(os.environ["REVIEWED_CRATES"]) / f"{name}-{VERSION}.crate"
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def credential(request):
    # Cargo asks for a read credential during preflight and the actual tarball
    # checksum just before upload. Never let it reuse the preflight response.
    registry = request["registry"]["index-url"]
    if registry not in ("https://github.com/rust-lang/crates.io-index", "sparse+https://index.crates.io/"):
        return {"Err": {"kind": "url-not-supported"}}
    if request["kind"] == "get" and request["operation"] == "read":
        token = "public-read"
    elif (request["kind"] == "get" and request["operation"] == "publish"
          and request["name"] in CRATES and request["vers"] == VERSION
          and request["cksum"] == reviewed_digest(request["name"])):
        token = os.environ["PUBLISH_TOKEN"]
    else:
        return {"Err": {"kind": "other", "message": "Cargo must publish the exact reviewed archive."}}
    return {"Ok": {"kind": "get", "token": token, "cache": "never", "operation_independent": False}}


def index_version(name):
    try:
        with urlopen(f"https://index.crates.io/ze/ro/{name}", timeout=30) as response:
            return next((row for row in map(json.loads, response) if row["vers"] == VERSION), None)
    except HTTPError as error:
        if error.code != 404:
            raise
        return None


if __name__ == "__main__":
    if sys.argv[1:] == ["--cargo-plugin"]:
        print(json.dumps({"v": [1]}), flush=True)
        for line in sys.stdin:
            print(json.dumps(credential(json.loads(line))), flush=True)
    else:
        if not os.environ.get("PUBLISH_TOKEN"):
            sys.exit("Missing crates.io publishing credential for this tag.")
        provider = str(Path(__file__).resolve())
        for crate in CRATES:
            present = index_version(crate)
            if present:
                if present["cksum"] != reviewed_digest(crate) or present["yanked"]:
                    sys.exit(f"{crate} {VERSION} on crates.io differs from the reviewed release.")
                print(f"{crate} {VERSION} already matches the reviewed archive.", flush=True)
                continue
            subprocess.run(["cargo", "publish", "--registry", "crates-io", "--locked", "-p", crate],
                           cwd="rust", env={**os.environ, "CARGO_REGISTRY_CREDENTIAL_PROVIDER": provider},
                           check=True)
