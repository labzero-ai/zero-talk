# Zero Talk for Python

`zero-talk`: Zero Talk 11M, an English text-to-speech model with about 11M parameters, for Python. The package is the
compiled engine with the model inside (about 8 MB per wheel): no PyTorch, no GPU, no download on first use, no network
access at runtime. Made for reading short English texts aloud where size and privacy matter. Not for long narration,
other languages or imitating real people.

Model page and live demo: https://labzero.ai/zero-talk-11m/

## Install and first audio

```sh
pip install zero-talk
```

```python
import zero_talk

voice = zero_talk.load_voice("labzero-ai/zero-talk-11m")
audio = voice.speak("Hello from a small voice.")
audio.save("hello.ogg")      # Ogg Opus, 24 kHz mono: 1.8 s of speech in 5 KB
```

Name the file `hello.wav` for an uncompressed WAV (17 times larger). Python has no playback call; open the file in any
player, or use the command line below.

Wheels exist for Python 3.9 and later on macOS (Apple silicon and Intel), Linux (x86-64 and ARM, manylinux) and Windows
(x86-64). On any other system, pip builds the source package, which needs a Rust toolchain (https://rustup.rs).

## Use

```python
audio.samples        # read-only float32 memoryview, mono, 24,000 samples per second, values from -1 to 1
audio.sample_rate    # 24000
audio.seconds
audio.to_wav()       # bytes, a WAV file
audio.to_ogg()       # bytes, Ogg Opus
numpy.asarray(audio.samples)   # needs numpy (pip install numpy); no copy
```

Piece by piece, while it is still being made:

```python
import numpy as np   # pip install numpy
import zero_talk

voice = zero_talk.load_voice("labzero-ai/zero-talk-11m")
chunks = [np.asarray(chunk.samples) for chunk in voice.stream("A longer text, read piece by piece.")]
samples = np.concatenate(chunks)   # float32, 24,000 samples per second: the same samples speak() returns
```

To save a stream's audio as a file, call `speak()` and `save()`: the audio is the same.

## API at a glance

| Call | Returns |
| --- | --- |
| `zero_talk.load_voice(id, model_dir=None)` | `Voice`. `id` is `"labzero-ai/zero-talk-11m"`, the only voice in this version; `zero_talk.voices()` lists them. `model_dir` reads the files from a folder instead. |
| `voice.speak(text)` | `Audio`: `samples`, `sample_rate`, `seconds`, `warnings`, `save(path)`, `to_wav()`, `to_ogg()` (bytes) |
| `voice.stream(text)` | iterator of chunks with `samples` |

## Text it reads

English, plain sentences. The front end rewrites written forms as the words a person would say before the voice reads them.
Examples of that rewriting step (digits are turned into words afterwards):

| Written | Rewritten as |
| --- | --- |
| `Elm Ave.` (more text follows) | `Elm Avenue` |
| `9:30 a.m.` | `9 thirty A M` |
| `800-555-0199` | digits, one by one |
| `2.75 kg`, `10 km` | `two point seven five kilograms`, `10 kilometers` |
| `18 °C` | `18 degrees Celsius` |
| `2026-11-05` | `November 5th twenty twenty-six` |
| `example.com` | `example dot com` |
| `€20` | `20 euros` |

A `St.` at the very end of a text is not expanded. Other scripts are skipped with a warning on `audio.warnings`.

## Command line

The package also installs the `zero-talk` command (the engine's own command line):

```sh
zero-talk "Hello there." -o hello.ogg         # Ogg Opus; a .wav name writes WAV
zero-talk "Hello there."                      # plays it through a player on your system (see below)
zero-talk --stream "Hello there." > out.ogg   # audio to a pipe while it renders
```

Playing without `-o` needs a player: `afplay` or `ffplay` on macOS, `paplay`, `pw-play`, `aplay` or `ffplay` on Linux,
PowerShell on Windows. Without one the command stops with `no audio player found (tried …); write a file instead: add -o OUT.ogg`.

## Where the model files come from

The ID names files compiled into the package, the same bytes as the model page lists (sizes and SHA-256 in
`zero_talk.model_info("labzero-ai/zero-talk-11m")`). A given package version always gives the same audio on the same system;
it differs between operating systems, system versions and processors by floating-point rounding. On macOS the wheel's matrix
products use Apple's Accelerate framework, which is about 3.5 times faster than the pure Rust code the other platforms use.

## Privacy

No text you give it and no audio it makes leaves the device. The package makes no network request at runtime.

## Speed and memory

Measured with Zero Talk 1.0 on 2026-11-09 (UTC), reading the 503-character text in
[`bench/text.txt`](https://github.com/labzero-ai/zero-talk/blob/main/bench/text.txt) (32.8 s of speech) with Python 3.12 on an
Apple M1 Pro (10 cores, 32 GB, macOS wheel): 10 times real time (seconds of speech made per second of compute), peak memory
about 230 MB, voice load 0.1 s. The pure Rust backend used on Linux, Windows and Intel Macs ran 2.7 times real time on the same
machine; other CPUs differ. Memory grows with the text: 3,280 s of speech from 50,300 characters took 321 s (10 times real time) and peaked at about 910 MB. `bench/python.py` in the repository
measures your own machine.

## Why this one

About 9 MB with the engine, against 68 to 341 MB for the files Pocket TTS and Kokoro-82M install (not counting PyTorch).
On 52 everyday sentences read back by a speech recogniser, the word error rate was 6.5% to 6.7% (95% intervals 4.1 to 9.6%),
level with Pocket TTS 18.2M and Kokoro-82M, whose intervals overlap (trial of 2026-11-06). It has one voice and reads
English only. More detail is in the [repository README](https://github.com/labzero-ai/zero-talk#why-this-one).

## What it does and does not do

English only, one voice, 24 kHz mono. Questions come out flat and exclamations do not come across; long sentences can lose
their rhythm; some S sounds have a slight lisp; "Hi, …" greetings can sound unnatural; some names are read oddly; there are
no tempo or pause controls. Short, plain sentences work best. The full list is on the model page.

Silent cases: most symbols and emoji are left out without a warning, whole numbers of more than 306 digits are dropped,
and a lone `.` speaks as 0.29 s of audio.

WAV and Ogg files from `to_wav()`, `to_ogg()` and `save()` carry an AI-generated label in their metadata: a comment in both
and, in Ogg, the IPTC digital source type `trainedAlgorithmicMedia`. It is a label, not a watermark: re-encoding or stripping
metadata removes it, and `audio.samples` carries none.

## FAQ

**Can it clone a voice or imitate a person?** No. It has one fixed voice and no way to pass a reference recording.

**Why Ogg Opus and not MP3?** Ogg Opus is an open codec with small files at this quality (24 kbit/s). Some old players cannot
open it; convert once with `ffmpeg -i hello.ogg hello.mp3`.

**Does it send my text anywhere?** No. See Privacy above.

**Where do I report a problem?** [GitHub issues](https://github.com/labzero-ai/zero-talk/issues).

## Errors and limits

Measured on 2026-11-06 with Python 3.12 and this package. Quoted texts are exact.

| Case | What happens |
| --- | --- |
| Empty text, spaces only, `...`, an emoji, or text with no letter the model can read (Chinese, for example) | `speak()` raises `ValueError: nothing to say: the text has no words the model can read; pass a sentence made of letters, for example "Hello."`. `stream()` raises the same error at the call, before the first chunk. |
| Steering tags (`[M2]`, `[breathy:1]`) | Not supported: removed before reading, and named in `audio.warnings`. Links `[label](target)` and `[aside]` are read as written. |
| A lone `.`, `?` or `!` | Speaks as 0.29 s of audio, with no warning. |
| Letters outside a to z (Cyrillic, Greek, Chinese) in text that has other words | Left out, and named in `audio.warnings`, for example `["1 word(s) could not be read and are left out: мир"]`. Text whose only readable parts are punctuation (`Привет, мир.`) speaks 0.46 s with that warning; `Привет мир` with no punctuation raises the error above. |
| Most symbols and emoji (`#`, `~`, `/`, `😀`) | Left out with no warning. Some symbols are read as words: `&` as "and", `%` as "percent", `+` as "plus", `=` as "equals", `@` as "at", `©` as "copyright". |
| A whole number of more than 306 digits | Left out with no warning; a 307-digit number leaves only its full stop. |
| Length | A text of 5,016 characters (96 sentences) and a text of 50,000 characters (8,908 words) speak in full, not truncated; one sentence of 11,263 characters with no full stop also speaks in full. The text is split into pieces of at most 510 phonemes (the units of sound the model reads); a longer sentence is split too, with no warning. |
| Unknown voice ID | `ValueError: no voice named 'labzero-ai/unknown'; the voices are: labzero-ai/zero-talk-11m. zero_talk.voices() lists them.` |
| `model_dir` without the model files | `FileNotFoundError: <folder>/model.bin: no such file; model_dir must hold model.bin and frontend.bin (zero_talk.model_info() gives their sizes and hashes)` |
| `model_dir` with a file of another kind | `ValueError: <folder>/model.bin: this is a 'frontend' file, not a model file` |
| `save()` with another extension (`.mp3`, none) | `ValueError: <path>: save() writes .wav (WAV) or .ogg (Ogg Opus); name the file with one of those extensions`. Nothing is written. `.WAV` in capitals works. |
| `save()` into a missing folder | `FileNotFoundError: <path>: No such file or directory; create the folder first` |
| Text that is not a string | `TypeError: 'int' object is not an instance of 'str'` |

Streaming: `stream()` yields a chunk for each piece as it is made; for a three-sentence text that is 11 chunks, the first of
55 samples (about 2 ms), and the chunks joined equal the samples of `speak()`.

A voice can be used from several threads. Calls are made one at a time, and the audio is the same as when they run one after
another. A `Stream` must be used by the thread that made it; used from another thread it raises an exception (a
`PanicException`, which is not a subclass of `Exception`).

## Licence

Apache-2.0, for the code and the model files. Zero Talk builds on Kokoro-82M (Apache-2.0). Third-party notices are in
[THIRD-PARTY-NOTICES.txt](https://github.com/labzero-ai/zero-talk/blob/main/python/THIRD-PARTY-NOTICES.txt).

Lab Zero · https://labzero.ai
