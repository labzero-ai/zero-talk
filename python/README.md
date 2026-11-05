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
| `voice.speak(text)` | `Audio`: `samples`, `sample_rate`, `seconds`, `save(path)`, `to_wav()`, `to_ogg()` (bytes) |
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

## Licence

Apache-2.0, for the code and the model files. Zero Talk builds on Kokoro-82M (Apache-2.0). Third-party notices are in
[THIRD-PARTY-NOTICES.txt](https://github.com/labzero-ai/zero-talk/blob/main/python/THIRD-PARTY-NOTICES.txt).

Lab Zero · https://labzero.ai
