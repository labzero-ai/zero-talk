# Zero Talk

Zero Talk 11M, an English text-to-speech model with about 11M parameters, and its engine in Rust: a library and the
`zero-talk` command. The model is compiled in, so a build speaks on its own: no Python, no GPU, no network access. Made for
reading short English texts aloud where size and privacy matter. Not for long narration, other languages or imitating real
people.

Model page and live demo: https://labzero.ai/zero-talk-11m/

| Crate | What it is |
| --- | --- |
| `zero-talk` | The engine and the `zero-talk` command. Embeds the model by default. |
| `zero-talk-model-11m` | The model and text front-end files only, versioned with the model. |

## Command line

`cargo install zero-talk` needs a Rust toolchain (https://rustup.rs) and compiles the engine locally, which takes a few minutes.

```sh
cargo install zero-talk

zero-talk "Hello there." -o hello.ogg     # Ogg Opus at 24 kbit/s, 24 kHz mono: 1.8 s of speech in 5 KB; a .wav name writes WAV
zero-talk "Hello there."                  # plays it while it renders, through a player on your system (see below)
zero-talk --stream "Hello there." > out.ogg
```

Playing without `-o` needs a player: `afplay` (or `ffplay` for `--stream`) on macOS, `paplay`, `pw-play`, `aplay` or `ffplay`
on Linux, PowerShell on Windows. Without one the command stops with `no audio player found (tried …); write a file instead: add -o OUT.ogg`.

## Library

```toml
[dependencies]
zero-talk = "1"
```

```rust
fn main() -> zero_talk::Result<()> {
    let voice = zero_talk::Voice::load("labzero-ai/zero-talk-11m")?;
    let audio = voice.speak("Hello from a small voice.")?;

    audio.save("hello.ogg")?;          // the extension picks the format: .ogg (Ogg Opus) or .wav
    let samples: &[f32] = audio.samples();   // mono, 24 kHz, values from -1 to 1
    assert!(!samples.is_empty());

    for chunk in voice.stream("A longer text, read piece by piece.")? {
        let chunk = chunk?;
        let samples: &[f32] = chunk.samples();
        println!("{} samples", samples.len());
    }
    Ok(())
}
```

## API at a glance

| Call | Returns |
| --- | --- |
| `Voice::load(id)` | `Result<Voice>`. `id` is `"labzero-ai/zero-talk-11m"`, the only voice in this version; `zero_talk::voices()` lists them. Needs the default `embed` feature. |
| `Voice::load_from_dir(id, dir)` | `Result<Voice>`, from a folder that holds `model.bin` and `frontend.bin` (works without `embed`) |
| `Voice::from_files(model, frontend)` | `Result<Voice>`, from the two files at any paths and names (works without `embed`) |
| `voice.speak(text)` | `Result<Audio>`: `samples()`, `sample_rate()`, `seconds()`, `warnings()`, `save(path)`, `encode(Format::Wav)` or `encode(Format::Ogg { bitrate: 64 })` (bytes; `Format::OGG` is Ogg at 24 kbit/s, bit rates 8 to 128) |
| `voice.stream(text)` | `Result<Stream>`: an iterator of `Result<Chunk>`, each with `samples()` |

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

Dates (`2026-11-05`, `11/05/2026`) are read month first unless a field is above 12. A `St.` at the very end of a text is not
expanded. Other scripts are skipped with a warning on `audio.warnings()`.

## Where the model files come from

`zero-talk-model-11m` holds the model and front-end files as byte slices, `zero_talk_model_11m::MODEL` and
`zero_talk_model_11m::FRONTEND`, with their sizes and SHA-256 hashes (`MODEL_BYTES`, `MODEL_SHA256`, `FRONTEND_BYTES`,
`FRONTEND_SHA256`), the same as the model page lists. The engine embeds them through its default `embed` feature, and
`Voice::load` finds them by ID. Keeping the data in its own crate means an engine update does not re-publish the model. To
ship your own copy instead:

```toml
zero-talk = { version = "1", default-features = false }
```

Without `embed`, `Voice::load` returns an error that says so; give the folder that holds the two files. To get them from the crate:

```rust
std::fs::create_dir_all("voice")?;
std::fs::write("voice/model.bin", zero_talk_model_11m::MODEL)?;       // add zero-talk-model-11m = "1" to Cargo.toml
std::fs::write("voice/frontend.bin", zero_talk_model_11m::FRONTEND)?;
let voice = zero_talk::Voice::load_from_dir("labzero-ai/zero-talk-11m", "voice")?;
```

`Voice::from_files(model, frontend)` takes the two files at any paths and names instead.

The same two files are in the Hugging Face repository `labzero-ai/zero-talk-11m` (https://huggingface.co/labzero-ai/zero-talk-11m).

## Privacy

No text you give it and no audio it makes leaves the device. The library and the command make no network request.

## Speed and memory

Measured with Zero Talk 1.0 on 2026-11-09 (UTC), reading the 503-character text in
[`bench/text.txt`](https://github.com/labzero-ai/zero-talk/blob/main/bench/text.txt) (32.8 s of speech) with a release build on
an Apple M1 Pro (10 cores, 32 GB):

| Backend | Speed | Peak memory |
| --- | --- | --- |
| macOS default (Apple's Accelerate library) | 10 times real time | about 230 MB |
| `--backend portable` (pure Rust; the only backend on Linux and Windows) | 2.7 times real time | about 200 MB |

"Real time" means seconds of speech made per second of compute. Memory grows with the text: 3,280 s of speech from 50,300
characters took 335 s and peaked at about 730 MB. Other CPUs differ; `zero-talk` prints its own speed after each file, and
`bench/` in the repository has the text and scripts to measure your own machine.

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

WAV and Ogg files from `encode()`, `save()` and the command carry an AI-generated label in their metadata: a
comment in both and, in Ogg, the IPTC digital source type `trainedAlgorithmicMedia`. It is a label, not a watermark:
re-encoding or stripping metadata removes it, and `audio.samples()` and `--format pcm` carry none.

The tests pass on macOS (Apple silicon), Linux (x86-64 and ARM) and Windows (x86-64). The audio is the same on every run on
one system; it differs between operating systems, system versions and processors by floating-point rounding.

## FAQ

**Can it clone a voice or imitate a person?** No. It has one fixed voice and no way to pass a reference recording.

**Why Ogg Opus and not MP3?** Ogg Opus is an open codec with small files at this quality (24 kbit/s). `--bitrate` sets it from
8 to 128 kbit/s. Some old players cannot open it; convert once with `ffmpeg -i hello.ogg hello.mp3`.

**Does it send my text anywhere?** No. See Privacy above.

**Where do I report a problem?** [GitHub issues](https://github.com/labzero-ai/zero-talk/issues).

## Errors and limits

Measured on 2026-11-06 with a release build of this crate. Quoted texts are exact. `speak()` and `stream()` return an
`Error` whose text is shown; the `zero-talk` command prints the same text after `zero-talk: ` and exits with status 2.

| Case | What happens |
| --- | --- |
| Empty text, spaces only, `...`, an emoji, or text with no letter the model can read (Chinese, for example) | `Err`: `nothing to say: the text has no words the model can read; pass a sentence made of letters, for example "Hello."`. `stream()` returns the same error. |
| Steering tags (`[M2]`, `[breathy:1]`) | Not supported: removed before reading, and named in `audio.warnings()`. Links `[label](target)` and `[aside]` are read as written. |
| A lone `.`, `?` or `!` | Speaks as 0.29 s of audio, with no warning. |
| Letters outside a to z (Cyrillic, Greek, Chinese) in text that has other words | Left out, and named by `audio.warnings()`, for example `1 word(s) could not be read and are left out: мир`. The command prints the warning on standard error. Text whose only readable parts are punctuation (`Привет, мир.`) speaks 0.46 s with that warning; `Привет мир` with no punctuation is the error above. |
| Most symbols and emoji (`#`, `~`, `/`, `😀`) | Left out with no warning. Some symbols are read as words: `&` as "and", `%` as "percent", `+` as "plus", `=` as "equals", `@` as "at", `©` as "copyright". |
| A whole number of more than 306 digits | Left out with no warning; a 307-digit number leaves only its full stop. |
| Length | A text of 5,016 characters (96 sentences) and a text of 50,000 characters (8,908 words) speak in full, not truncated. The text is split into pieces of at most 510 phonemes (the units of sound the model reads); a longer sentence is split too, with no warning. |
| Unknown voice ID | `Err`: `no voice named 'labzero-ai/unknown'; the voices are: labzero-ai/zero-talk-11m` |
| `Voice::load_from_dir` with a file missing | `Err`: `<folder>/frontend.bin: not found; the folder must hold model.bin and frontend.bin (the data folder of the zero-talk-model-11m crate)` |
| `Voice::from_files` with the files missing | `Err`: `<folder>/frontend.bin: No such file or directory` (the front-end file is opened first) |
| `Voice::from_files` with a file of another kind | `Err`: `<folder>/model.bin: this is a 'frontend' file, not a model file` |
| `save()` with another extension (`.mp3`, none) | `Err`: `<path>: save() writes .wav (WAV) or .ogg (Ogg Opus); name the file with one of those extensions`. Nothing is written. `.WAV` in capitals works. |
| `encode(Format::Ogg { bitrate })` with a bit rate outside 8 to 128 | `Err`: `bitrate must be a whole number of kbit/s from 8 to 128, not 999` |
| `save()` into a missing folder | `Err`: `<path>: No such file or directory; create the folder first` |
| Command options | `zero-talk: --bitrate needs a whole number of kbit/s from 8 to 128, not '999'`; `zero-talk: x.mp3: the output file must end in .ogg (Ogg Opus) or .wav (WAV); MP3 is not built`; `--format` takes `ogg`, `wav` or `pcm`. |

Streaming: `stream()` yields one `Result<Chunk>` for each piece as it is made; for a three-sentence text that is 11 chunks,
the first of 55 samples (about 2 ms), and the chunks joined equal the samples of `speak()`.

## Licence

Apache-2.0, for the code and the model files. Zero Talk builds on Kokoro-82M (Apache-2.0). Third-party notices are in
[THIRD-PARTY-NOTICES.md](https://github.com/labzero-ai/zero-talk/blob/main/rust/THIRD-PARTY-NOTICES.md).

Lab Zero · https://labzero.ai
