# Working on zero-talk

Zero Talk 11M, an English text-to-speech model with about 11M parameters, and its packages:

- `rust/`: a cargo workspace with the crates `zero-talk` (the engine and the `zero-talk` command) and
  `zero-talk-model-11m` (the model and text front-end files, embedded by default).
- `python/`: the `zero-talk` wheel (PyO3 and maturin, abi3 for Python 3.9 and later), built from `rust/`.
- `js/`: the npm package `@labzero-ai/zero-talk` (WebGPU with a WebAssembly fallback, browsers and Node.js 22+).
- `huggingface/`: the model card and the script that assembles the Hugging Face upload.

## Checks

- `rust/`: `cargo test --release --workspace` (unit tests, API tests and doctests).
- `python/`: `maturin develop --release`, then `python -m unittest discover -s tests`.
- `js/`: `npm test`.

CI (`.github/workflows/ci.yml`) runs them on Linux (x86-64, ARM), Windows and macOS. Releases: `RELEASING.md`.

The npm package writes byte-identical audio everywhere (WebAssembly). Native builds give the same audio on every run on
one system and differ between systems and system versions by floating-point rounding, so byte checks against
fixed hashes run on one platform only.

## API and documentation

Written for AI agents first: most code that uses these packages is written by agents, so choose what an agent would
guess and use correctly.

- Explicit, conventional calls: load by model ID (`labzero-ai/zero-talk-11m`), as in `from_pretrained`.
- One return type per call; errors that name the fix.
- The same API shape in Rust, Python and JavaScript (`load_voice` / `loadVoice` / `Voice::load`, `speak`, `stream`, one `Audio` type).
- README examples complete enough to copy, and checked against the real API.
- Third-party notices are copied verbatim, typos included.
