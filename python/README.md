# Zero Talk for Python

`zero-talk`: Zero Talk 11M, an English text-to-speech model with about 11M parameters, for Python. The package is the
compiled engine with the model inside (about 8 MB per wheel): no PyTorch, no GPU, no download on first use, no network
access at runtime.

Model page and live demo: https://labzero.ai/zero-talk-11m/

## Install

```sh
pip install zero-talk
```

## Use

```python
import zero_talk

voice = zero_talk.load_voice("labzero-ai/zero-talk-11m")
audio = voice.speak("Hello from a small voice.")

audio.save("hello.wav")      # the extension picks the format: .wav or .ogg (Ogg Opus)
audio.samples                # read-only float32 memoryview, mono; numpy.asarray(audio.samples) needs no copy
audio.sample_rate            # 24000
audio.seconds
```

WAV and Ogg files carry an AI-generated marker in their metadata.

Piece by piece, while it is still being made:

```python
import zero_talk

voice = zero_talk.load_voice("labzero-ai/zero-talk-11m")
for chunk in voice.stream("A longer text, read piece by piece."):
    chunk.samples
```

## API at a glance

| Call | Returns |
| --- | --- |
| `zero_talk.load_voice(id, model_dir=None)` | `Voice`. `id` is `"labzero-ai/zero-talk-11m"`, the only voice in this version; `zero_talk.voices()` lists them. `model_dir` reads the files from a folder instead. |
| `voice.speak(text)` | `Audio`: `samples`, `sample_rate`, `seconds`, `warnings`, `save(path)`, `to_wav()`, `to_ogg()` (bytes) |
| `voice.stream(text)` | iterator of chunks with `samples` |

Text: English, plain sentences; numbers, dates and prices written as digits are read out. Other scripts are skipped with a
warning on `audio.warnings`.

## Command line

The package also installs the `zero-talk` command (the engine's own command line):

```sh
zero-talk "Hello there." -o hello.ogg     # Ogg Opus; a .wav name writes WAV
zero-talk "Hello there."                  # plays it
zero-talk --stream "Hello there." > out.ogg   # audio to a pipe while it renders
```

## Where the model files come from

The ID names files compiled into the package, the same bytes as the model page lists (sizes and SHA-256 in
`zero_talk.model_info("labzero-ai/zero-talk-11m")`). A given package version always gives
the same audio on the same system; it differs between operating systems, system versions and processors by
floating-point rounding.

## What it does and does not do

English only, one voice, 24 kHz mono. Questions come out flat, and long sentences can lose their rhythm; short, plain
sentences work best. The full list is on the model page.

Wheels for Python 3.9 and later on macOS (Apple silicon), Linux (x86-64 and ARM, manylinux) and Windows (x86-64); the
tests pass on each.

## Errors and limits

Measured on 2026-11-06 with Python 3.12 and this package. Quoted texts are exact.

| Case | What happens |
| --- | --- |
| Empty text, spaces only, `...`, an emoji, or text with no letter the model can read (Chinese, for example) | `speak()` raises `ValueError: nothing to say: the text has no words the model can read; pass a sentence made of letters, for example "Hello."`. `stream()` raises the same error at the call, before the first chunk. |
| A lone `.`, `?` or `!` | Speaks as 0.29 s of audio, with no warning. |
| Letters outside a to z (Cyrillic, Greek, Chinese) in text that has other words | Left out, and named in `audio.warnings`, for example `["1 word(s) could not be read and are left out: мир"]`. Text whose only readable parts are punctuation (`Привет, мир.`) speaks 0.46 s with that warning; `Привет мир` with no punctuation raises the error above. |
| Most symbols and emoji (`#`, `~`, `/`, `😀`) | Left out with no warning. Some symbols are read as words: `&` as "and", `%` as "percent", `+` as "plus", `=` as "equals", `@` as "at", `©` as "copyright". |
| A whole number of more than 306 digits | Left out with no warning; a 307-digit number leaves only its full stop. |
| Length | A text of 5,016 characters (96 sentences) speaks in full: 306 s of speech in 96 s. A text of 50,000 characters (8,908 words) also speaks in full: 3,049 s of speech in 972 s, peak memory about 966 MB. One sentence of 11,263 characters with no full stop speaks in full (644 s of speech). The text is split into pieces of at most 510 phonemes; a longer sentence is split too, with no warning. |
| Unknown voice ID | `ValueError: no voice named 'labzero-ai/unknown'; the voices are: labzero-ai/zero-talk-11m. zero_talk.voices() lists them.` |
| `model_dir` without the model files | `FileNotFoundError: <folder>/model.bin: no such file; model_dir must hold model.bin and frontend.bin (zero_talk.model_info() gives their sizes and hashes)` |
| `model_dir` with a file of another kind | `ValueError: <folder>/model.bin: this is a 'frontend' file, not a model file` |
| `save()` with another extension (`.mp3`, none) | `ValueError: <path>: save() writes .wav (WAV) or .ogg (Ogg Opus); name the file with one of those extensions`. Nothing is written. `.WAV` in capitals works. |
| `save()` into a missing folder | `FileNotFoundError: No such file or directory (os error 2)`. This message does not name the file. |
| Text that is not a string | `TypeError: 'int' object is not an instance of 'str'` |

Streaming: `stream()` yields a chunk for each piece as it is made; for a three-sentence text that is 11 chunks, the first of
55 samples (about 2 ms), and the chunks joined equal the samples of `speak()`.

A voice can be used from several threads. Calls are made one at a time, and the audio is the same as when they run one after
another. A `Stream` must be used in the thread that made it; another thread gets a `PanicException`.

## Licence

Apache-2.0, for the code and the model files. Zero Talk builds on Kokoro-82M (Apache-2.0). Third-party notices are in
[THIRD-PARTY-NOTICES.txt](https://github.com/labzero-ai/zero-talk/blob/main/python/THIRD-PARTY-NOTICES.txt).

Lab Zero · https://labzero.ai
