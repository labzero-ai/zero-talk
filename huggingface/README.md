---
license: apache-2.0
language:
  - en
pipeline_tag: text-to-speech
base_model: hexgrad/Kokoro-82M
tags:
  - on-device
  - webgpu
  - webassembly
  - rust
  - streaming
---

# Zero Talk 11M

An English text-to-speech model with about 11M parameters. It runs on the user's device, in a browser, Node.js, Python or Rust, and no text or audio leaves it. Packages for each language carry the model; the browser demo is linked below.

[Live demo](https://labzero.ai/zero-talk-11m/)

## Use

The files use Lab Zero's own format, read by the `zero-talk` engine. They do not load in PyTorch or `transformers`.
Use `@labzero-ai/zero-talk` for JavaScript, `zero-talk` for Python, or the `zero-talk` Rust crate instead.
The Python and Rust packages also install a `zero-talk` command.
The packages include the model files. The model ID selects those files; it does not download them from Hugging Face.
These examples use the 1.0.0 package API.
WAV and Ogg files written by the packages carry an AI-generated label in their metadata (a comment in both and, in Ogg, the IPTC digital source type `trainedAlgorithmicMedia`). It is a label, not a watermark: re-encoding or stripping metadata removes it, and raw samples carry none.

### JavaScript

```sh
npm install @labzero-ai/zero-talk
```

This package uses ES modules. In Node.js, use a `.mjs` file or set `"type": "module"` in your `package.json`.
CommonJS works from a `.cjs` file (or any `.js` file in a project without `"type": "module"`):

```js
const { loadVoice } = require('@labzero-ai/zero-talk');

(async () => {
  const voice = await loadVoice('labzero-ai/zero-talk-11m');
  const audio = await voice.speak('Hello from a small voice.');
  await audio.save('hello.ogg');
})();
```

```js
import { loadVoice } from '@labzero-ai/zero-talk';

const voice = await loadVoice('labzero-ai/zero-talk-11m');
const audio = await voice.speak('Hello from a small voice.');

await audio.save('hello.ogg');   // Node.js: Ogg Opus, 1.8 s of speech in 5 KB; name it hello.wav for WAV (17 times larger)

audio.samples      // Float32Array, mono
audio.sampleRate   // 24000
audio.seconds
audio.toWav()      // Uint8Array, a WAV file (synchronous)
await audio.toOgg()  // Uint8Array, Ogg Opus (smaller; a promise)
await audio.play()   // browser only
```

In a browser there is no file system: `save()` rejects there. Use `toWav()` or `await audio.toOgg()` and offer the bytes
as a download.

Play while it is still being made, or take it piece by piece:

```js
await voice.play('This starts playing before the sentence is finished.');   // browser only

const parts = [];
for await (const chunk of voice.stream('A longer text, read piece by piece.')) {
  if (chunk.restart) parts.length = 0;   // the GPU engine failed: the text starts again on WebAssembly
  else parts.push(chunk.samples);        // Float32Array
}
```

In Node.js the engine is WebAssembly, which makes the whole text first: `stream()` yields one chunk there, and the audio is
the same as `speak()` returns. In a browser with WebGPU, chunks arrive while the text is still being made.

### Python

```sh
pip install zero-talk
```

```python
import zero_talk

voice = zero_talk.load_voice("labzero-ai/zero-talk-11m")
audio = voice.speak("Hello from a small voice.")

audio.save("hello.ogg")      # Ogg Opus, 1.8 s of speech in 5 KB; name it hello.wav for WAV (17 times larger)
audio.samples                # read-only float32 memoryview, mono; numpy.asarray(audio.samples) needs no copy
audio.sample_rate            # 24000
audio.seconds
```

Piece by piece, while it is still being made:

```python
import zero_talk

voice = zero_talk.load_voice("labzero-ai/zero-talk-11m")
for chunk in voice.stream("A longer text, read piece by piece."):
    chunk.samples
```

### Rust

```sh
cargo install zero-talk

zero-talk "Hello there." -o hello.ogg     # Ogg Opus at 24 kbps; a .wav name writes WAV
zero-talk "Hello there."                  # plays it while it renders, through a player on your system (see the Rust README)
zero-talk --stream "Hello there." > out.ogg
```

```toml
[dependencies]
zero-talk = "1"
```

```rust
fn main() -> zero_talk::Result<()> {
    let voice = zero_talk::Voice::load("labzero-ai/zero-talk-11m")?;
    let audio = voice.speak("Hello from a small voice.")?;

    audio.save("hello.ogg")?;          // the extension picks the format: .ogg (Ogg Opus) or .wav
    let samples: &[f32] = audio.samples();   // mono, 24 kHz

    for chunk in voice.stream("A longer text, read piece by piece.")? {
        let chunk = chunk?;
        let samples: &[f32] = chunk.samples();
        println!("{} samples", samples.len());
    }
    Ok(())
}
```

### Errors and limits

Measured on 2026-11-06 with the three packages. Exact error texts are in each package's README.

- Empty text, text with only spaces or punctuation (`...`), an emoji, or text with no letter the model can read: an error, `nothing to say: the text has no words the model can read; pass a sentence made of letters, for example "Hello."`. A lone `.`, `?` or `!` speaks as 0.29 s of audio.
- Letters outside a to z (Cyrillic, Greek, Chinese) are left out and named in the warnings. Most symbols and emoji are left out with no warning (some, such as `&` and `%`, are read as "and" and "percent"), and so are whole numbers of more than 306 digits.
- Length: a text of 5,016 characters, a text of 50,000 characters (3,049 s of speech) and one sentence of 11,263 characters with no full stop all speak in full, not truncated. Speech is made in pieces of at most 510 phonemes.
- An unknown voice ID, an unknown engine (JavaScript) or a file name without `.wav` or `.ogg` is an error that names the valid choices.
- Streaming: JavaScript in Node.js yields the whole text as one chunk; Python and Rust yield a chunk for each piece as it is made.

## Why this one

- **Small:** about 9 MB with the engine, against 68 MB (Pocket TTS 6.9M), 116 MB (Pocket TTS 18.2M) and 341 MB (Kokoro-82M) for
  the files those systems install, not counting the PyTorch they need (trial of 2026-11-06).
- **Clear enough to read back:** 52 everyday sentences (numbers, dates, prices, abbreviations, questions) played to a speech
  recogniser: word error rate 6.5% with faster-whisper `base.en` (95% interval 4.1 to 9.3%) and 6.7% with `small.en` (4.2 to
  9.6%). In the same trial Pocket TTS 18.2M scored 9.4% and 6.8%, and Kokoro-82M 6.0% and 5.4%. The intervals overlap, so the
  set cannot tell the three apart; it does not measure naturalness.
- **Private:** no text or audio leaves the device. Python, Rust and Node.js make no network request; a browser fetches the
  package's own files (about 9 MB) from whichever host serves them.
- **Given up:** one voice, English only, no tempo or pause control. Pocket TTS ran faster on the same CPU in our trial.

## Speed and memory

Measured with Zero Talk 1.0 on 2026-11-09 (UTC), reading a 503-character text (32.8 s of speech) on an Apple M1 Pro (10 cores,
32 GB). "Real time" means seconds of speech made per second of compute.

| Package | Speed | Peak memory |
| --- | --- | --- |
| Rust and Python on macOS (Apple's Accelerate library) | 10 times real time | about 230 MB |
| Rust, pure-Rust backend (Linux, Windows, Intel Macs) | 2.7 times real time | about 200 MB |
| Node.js (WebAssembly) | 1.5 times real time | about 215 MB |

In a browser with WebGPU the speed depends on the GPU; the demo shows the figure for your device.

## FAQ

**Can it clone a voice or imitate a person?** No. It has one fixed voice and no way to pass a reference recording.

**Why Ogg Opus and not MP3?** Ogg Opus is an open codec with small files at this quality (24 kbit/s). Some old players cannot
open it; convert once with `ffmpeg -i hello.ogg hello.mp3`.

**Where do I report a problem?** [GitHub issues](https://github.com/labzero-ai/zero-talk/issues).

## Files

Sizes and SHA-256 values below are from the release's `model.json`. Licence files keep their published bytes;
only their names change here. The manifest uses the original `.txt` names and links to the published model page.

| File here | Name in `model.json` | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| [model.bin](model.bin) | `model.bin` | 7053248 | `f0e885226e5fab84d9a800926479e420fc19161a5b4b435eb39d4ee102431ac5` |
| [frontend.bin](frontend.bin) | `frontend.bin` | 816248 | `5ade509c060811a5fe9525ac554d6edc2229c2fca66581e093463b9d9b8eaffd` |
| [LICENSE](LICENSE) | `LICENSE.txt` | 11357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| [NOTICE](NOTICE) | `NOTICE.txt` | 598 | `8665ac1b7a2ef970f15a20e86a9638f94e43e3aa5cface37b3be1599864a4025` |
| [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) | `THIRD-PARTY-NOTICES.txt` | 13670 | `6ae7b943ff9e925372b650172efefeffeacaf59051cd88ba52f394a2c816441b` |

[model.json](model.json) records the release facts and checksums. `README.md` is this card. Neither document has a
checksum entry in the manifest. The manifest also lists `engine.wasm`, the browser engine; that file ships in the
JavaScript package and is not included in this model repository.

## Specifications and limits

The specifications below describe the browser demo, as on the [model page](https://labzero.ai/zero-talk-11m/).

| Specification | Value |
| --- | --- |
| Parameters | about 11M (10.8M, plus about 94k for the voice) |
| Voice | English, one voice (Kokoro's af_heart) |
| Audio | 24 kHz, mono; the demo saves Ogg Opus |
| Download | about 9 MB, fetched only when you tap |
| Runs on | this device, in a background thread: WebGPU where the browser offers it on a secure page, otherwise WebAssembly. A GPU is not required. |
| Tested in | Chromium on an Apple-silicon laptop and Safari 26 on an iPhone 17 Pro, with both engines. Other browsers and phones are untested. |
| Licence | Apache-2.0, for the model files and the engine ([licence text](LICENSE) · [NOTICE](NOTICE)) |
| Version | v1 |

### Intended use and limits

Made for reading short English texts aloud where size and privacy matter. Not for long narration, other languages, or imitating real people. Short, plain sentences work best. We judged the limits below by listening, in a small number of trials, not with many listeners.

- One voice, English only.
- Questions come out flat, and exclamations do not come across.
- Some S sounds have a slight lisp.
- Long sentences can lose their rhythm.
- Sentences that open with a greeting and a comma ("Hi, …") can sound unnatural, and very short ones ("Hi.") can sound slightly cut off at the start.
- Some names are read oddly. Accented Latin words are read in plain spelling; other scripts are left out with a warning.
- No controls for tempo or pauses.
- The two browser engines give close but not identical audio.

### What it does not do

Not in this version:

- More than one voice, or control over delivery style and emphasis.
- Voice cloning: one fixed voice, and no way to pass a reference recording.
- Tempo and pause controls.
- Natural intonation for questions: they come out flat.

Not tested:

- Firefox, Safari on macOS, Android and mid-range phones.

Not possible with this model:

- Other languages: it reads English only.

## Licence and credits

Zero Talk, its model files and its engine, is licensed under the Apache License 2.0. Third-party parts keep their own licences, listed in the [notices](THIRD-PARTY-NOTICES).

Zero Talk builds on Kokoro-82M (Apache-2.0), whose author is hexgrad. Its voice is Kokoro's af_heart. Its sound generator started as Kokoro's own, was compressed to about a quarter of its parameters, then retrained to stream. The network that turns text into sound learned from Kokoro as its teacher.

The streaming design, the 4-bit compression, the Rust engine and its WebGPU version are Lab Zero's work. The text rules are ported from misaki (Apache-2.0); the pronunciation dictionary and the small network that guesses unknown words were made by Lab Zero from misaki's dictionaries, with espeak-ng among the tools used (see the [notices](THIRD-PARTY-NOTICES)). The text front end also uses data from spaCy and en_core_web_sm (MIT), LibriSpeech transcripts (CC BY 4.0) and Unicode 15.0.0.

[LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) contain the licence text and third-party attributions, copied verbatim from the release files. References to `.txt` names inside the notices retain the original spelling.

## Links

[Lab Zero](https://labzero.ai/) · [GitHub](https://github.com/labzero-ai/zero-talk) · [X](https://x.com/labzeroai)
