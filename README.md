# Zero Talk

Zero Talk 11M is an English text-to-speech model with about 11M parameters that runs on the user's device: a packaged
engine and one voice, about 9 MB in total, for Python, Rust, Node.js and browsers. No server, no API key, no PyTorch.

```sh
pip install zero-talk
python -c "import zero_talk; zero_talk.load_voice('labzero-ai/zero-talk-11m').speak('Hello from a small voice.').save('hello.ogg')"
```

That writes `hello.ogg` (Ogg Opus, 24 kHz mono, 1.8 s of speech in 5 KB). Use `hello.wav` for an uncompressed file, which is 17
times larger. Live demo and model page: https://labzero.ai/zero-talk-11m/

| Folder | Package | For |
| --- | --- | --- |
| [`python/`](python/) | `zero-talk` on PyPI | Python |
| [`js/`](js/) | `@labzero-ai/zero-talk` on npm | Browsers and Node.js |
| [`rust/`](rust/) | `zero-talk`, `zero-talk-model-11m` on crates.io | Rust, and the `zero-talk` command |

Every package loads the model by its ID, `labzero-ai/zero-talk-11m`. The model ships inside each package.

## Why this one

- **Small.** About 9 MB with the engine, against 68 MB (Pocket TTS 6.9M), 116 MB (Pocket TTS 18.2M) and 341 MB (Kokoro-82M)
  for the files those systems install, not counting the PyTorch they need (our trial, 2026-11-06).
- **Clear enough to read back.** 52 everyday sentences (numbers, dates, prices, abbreviations, questions), played to a
  speech recogniser: word error rate 6.5% with faster-whisper `base.en` (95% interval 4.1 to 9.3%) and 6.7% with `small.en`
  (4.2 to 9.6%). In the same trial Pocket TTS 18.2M scored 9.4% and 6.8%, and Kokoro-82M 6.0% and 5.4%. The 95% intervals overlap. These scores measure recogniser errors on this set.
- **Private.** No text or audio leaves the device. Python, Rust and Node.js load the bundled model locally. A browser fetches the
  package's own files, nothing else (see [Privacy](#privacy)).
- **Open.** Apache-2.0 for the engine and the model files.

What you give up: one voice (the `af_heart` voice of Kokoro-82M, the model Zero Talk learned from), English only, no
tempo or pause control, flat questions. Smaller voices built for CPU speed exist: Pocket TTS ran faster on the same CPU in
our trial.

## Speed and memory

Measured with Zero Talk 1.0 on 2026-11-09 (UTC), reading the 503-character text in [`bench/`](bench/) (32.8 s of speech) on
an Apple M1 Pro (10 cores, 32 GB). "Real time" means seconds of speech made per second of compute.

| Package | Speed | Peak memory | Voice load |
| --- | --- | --- | --- |
| Rust and Python on macOS (Apple's Accelerate library) | 10 times real time | about 230 MB | 0.1 s |
| Rust, pure-Rust backend (Linux, Windows; `--backend portable` on macOS) | 2.7 times real time | about 200 MB | 0.1 s |
| Node.js (WebAssembly) | 1.5 times real time | about 215 MB | 0.2 s |

Memory grows with the length of the text: 3,280 s of speech from 50,300 characters peaked at about 730 MB in Rust. In a
browser with WebGPU the speed depends on the GPU; the demo shows the figure for your device. Other CPUs differ: `bench/` has
the text and the scripts to measure your own machine.

## Privacy

No text you give it and no audio it makes leaves the device. Python and Rust read the files locally. Node.js reads the bundled files locally unless you supply an HTTP address. In a browser the
only requests are for the package's own files (engine, model and text front end, about 9 MB), made to whichever host serves
them: your server, or a CDN such as jsDelivr if you import the package from there. That host sees an ordinary file request.

## FAQ

**Can it clone a voice or imitate a person?** No. It has one fixed voice and no way to pass a reference recording.

**Other languages or more voices?** Not in 1.0: English and one voice.

**Why Ogg Opus and not MP3?** Ogg Opus is an open codec that gives small files at the quality of speech here (24 kbit/s).
Some old players cannot open it; convert once with `ffmpeg -i hello.ogg hello.mp3`.

**Is the audio marked as AI-generated?** WAV and Ogg files carry a label in their metadata: a comment in both, and in Ogg also
the IPTC digital source type `trainedAlgorithmicMedia`. It is a label, not a watermark: re-encoding or stripping metadata
removes it, and raw samples (`audio.samples`, `--format pcm`) carry none.

**Where do I report a problem?** [GitHub issues](https://github.com/labzero-ai/zero-talk/issues).

Apache-2.0. Zero Talk builds on Kokoro-82M (Apache-2.0); third-party notices are in each package.

Lab Zero · https://labzero.ai
