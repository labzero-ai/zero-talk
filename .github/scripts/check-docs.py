#!/usr/bin/env python3
"""Fails when a README states a fact that the code or the release set-up no longer holds.

Run from the repository root: python3 .github/scripts/check-docs.py
Each check names the file to edit. Add a check when a README fact has gone stale once.
"""
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
problems = []


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def fail(path, message):
    problems.append(f"{path}: {message}")


version = json.loads(read("js/package.json"))["version"]
major_minor = ".".join(version.split(".")[:2])
model = json.loads(read("js/dist/model.json"))
readmes = ["README.md", "js/README.md", "python/README.md", "rust/README.md",
           "rust/crates/zero-talk-model-11m/README.md", "huggingface/README.md"]

# 1. A CDN link names the exact published version, never a floating range such as @1.
for path in readmes:
    for found in re.findall(r"@labzero-ai/zero-talk@([^/\s'\"]+)/", read(path)):
        if found != version:
            fail(path, f"CDN link pins @{found}, the package version is {version}")

# 2. Install lines and the measured-figures lines name this release line.
for path in readmes:
    for found in re.findall(r'zero-talk = "(\d+)"', read(path)):
        if found != version.split(".")[0]:
            fail(path, f'Cargo line says zero-talk = "{found}", the major version is {version.split(".")[0]}')
    for found in re.findall(r"[Mm]easured with Zero Talk (\d+\.\d+)", read(path)):
        if found != major_minor:
            fail(path, f"figures are measured with {found}, the package is {version}: measure again (bench/) and update the line")

# 3. "about N MB" for the package equals the sum of the files the model page lists (engine, model, front end).
total_mb = sum(f["bytes"] for f in model["files"] if f["name"] in ("engine.wasm", "model.bin", "frontend.bin")) / 1e6
for path in ("README.md", "js/README.md", "huggingface/README.md"):
    for found in re.findall(r"[Aa]bout (\d+(?:\.\d+)?) MB(?= in total| with the engine|\)|, fetched)", read(path)):
        if abs(float(found) - total_mb) > 0.6:
            fail(path, f"says about {found} MB; model.json lists {total_mb:.1f} MB in total")

# 4. The Python README lists every wheel the release workflow builds, and no other.
targets = re.findall(r"target:\s*(\S+)", read(".github/workflows/release.yml"))
words = {"aarch64-apple-darwin": "Apple silicon", "x86_64-apple-darwin": "Intel",
         "aarch64-unknown-linux-gnu": "ARM", "x86_64-unknown-linux-gnu": "x86-64",
         "x86_64-pc-windows-msvc": "Windows"}
python_readme = read("python/README.md")
for target in targets:
    if target in words and words[target] not in python_readme:
        fail("python/README.md", f"the release workflow builds {target} but the README does not mention '{words[target]}'")
if "x86_64-apple-darwin" not in targets and "Intel" in python_readme:
    fail("python/README.md", "mentions an Intel Mac wheel that the release workflow does not build")

# 5. Minimum versions: README text equals what the manifests declare.
node_min = re.search(r"(\d+)", json.loads(read("js/package.json"))["engines"]["node"]).group(1)
if f"Node.js {node_min} or later" not in read("js/README.md"):
    fail("js/README.md", f"package.json needs Node.js {node_min}; the README must say 'Node.js {node_min} or later'")
py_min = re.search(r">=\s*(3\.\d+)", tomllib.loads(read("python/pyproject.toml"))["project"]["requires-python"]).group(1)
if f"Python {py_min} and later" not in python_readme:
    fail("python/README.md", f"pyproject.toml needs Python {py_min}; the README must say 'Python {py_min} and later'")

# 6. The model ID in every README is the one the package serves.
for path in readmes:
    for found in re.findall(r"labzero-ai/(zero-talk-11m[\w-]*)", read(path)):
        if found != "zero-talk-11m":
            fail(path, f"unknown model ID labzero-ai/{found}")
if model["id"] != "labzero-ai/zero-talk-11m":
    fail("js/dist/model.json", f"id is {model['id']}")

if problems:
    print("README facts out of date:", *problems, sep="\n  ")
    sys.exit(1)
print(f"README facts match version {version}.")
