# Zero Talk

Zero Talk 11M, an English text-to-speech model with about 11M parameters, and its engine in Rust: a library and the
`zero-talk` command. The model is compiled in, so a build speaks on its own: no Python, no GPU, no network access.

Model page and live demo: https://labzero.ai/zero-talk-11m/

| Crate | What it is |
| --- | --- |
| `zero-talk` | The engine and the `zero-talk` command. Embeds the model by default. |
| `zero-talk-model-11m` | The model and text front-end files only, versioned with the model. |

## Command line

```sh
cargo install zero-talk

zero-talk "Hello there." -o hello.ogg     # Ogg Opus at 24 kbps; a .wav name writes WAV
zero-talk "Hello there."                  # plays it while it renders (ffplay, paplay or aplay)
zero-talk --stream "Hello there." > out.ogg
```

WAV and Ogg files carry an AI-generated marker in their metadata.

## Library

```toml
[dependencies]
zero-talk = "1"
```

```rust
fn main() -> zero_talk::Result<()> {
    let voice = zero_talk::Voice::load("labzero-ai/zero-talk-11m")?;
    let audio = voice.speak("Hello from a small voice.")?;

    audio.save("hello.wav")?;          // the extension picks the format: .wav or .ogg (Ogg Opus)
    let samples: &[f32] = audio.samples();   // mono, 24 kHz
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
| `Voice::from_files(model, frontend)` | `Result<Voice>`, from model files on disk (works without `embed`) |
| `voice.speak(text)` | `Result<Audio>`: `samples()`, `sample_rate()`, `seconds()`, `save(path)`, `to_wav()`, `to_ogg()` |
| `voice.stream(text)` | `Result<Stream>`: an iterator of `Result<Chunk>`, each with `samples()` |

Text: English, plain sentences; numbers, dates (`2026-11-05`, `11/05/2026`, month first unless a field is above 12) and prices written as digits are read out. Other scripts are skipped with a
warning on `audio.warnings()`.

## Where the model files come from

`zero-talk-model-11m` holds the model and front-end files, the same bytes as the model page lists (sizes and SHA-256 in
`model.json`). The engine embeds them through its default `embed` feature, and `Voice::load` finds them by ID. Keeping
the data in its own crate means an engine update does not re-publish the model. To ship your own copy instead:

```toml
zero-talk = { version = "1", default-features = false }
```

Without `embed`, `Voice::load` returns an error that says so; give the two files yourself:

```rust
let voice = zero_talk::Voice::from_files("model.bin", "frontend.bin")?;
```

`model.bin` and `frontend.bin` are the files in the `data/` folder of the `zero-talk-model-11m` crate, and in the Hugging Face
repository `labzero-ai/zero-talk-11m`.

## What it does and does not do

English only, one voice, 24 kHz mono. Questions come out flat, and long sentences can lose their rhythm; short, plain
sentences work best. The full list is on the model page.

On macOS the matrix products run on Apple's Accelerate framework by default (about 3 to 4 times faster than the pure Rust code); `zero-talk --backend portable` selects the pure Rust backend. The tests pass on macOS (Apple silicon), Linux (x86-64 and ARM) and Windows (x86-64). The audio is the same on every
run on one system; it differs between operating systems, system versions and processors by floating-point rounding.

## Errors and limits

Measured on 2026-11-06 with a release build of this crate. Quoted texts are exact. `speak()` and `stream()` return an
`Error` whose text is shown; the `zero-talk` command prints the same text after `zero-talk: ` and exits with status 2.

| Case | What happens |
| --- | --- |
| Empty text, spaces only, `...`, an emoji, or text with no letter the model can read (Chinese, for example) | `Err`: `nothing to say: the text has no words the model can read; pass a sentence made of letters, for example "Hello."`. `stream()` returns the same error. |
| Steering tags (`[M2]`, `[breathy:1]`) | Removed before reading, and named in `audio.warnings()`: `steering is not supported in this build: removed 1 tag(s) from the text: [M2]`. Links `[label](target)` and `[aside]` are read as written. |
| A lone `.`, `?` or `!` | Speaks as 0.29 s of audio, with no warning. |
| Letters outside a to z (Cyrillic, Greek, Chinese) in text that has other words | Left out, and named by `audio.warnings()`, for example `1 word(s) could not be read and are left out: мир`. The command prints the warning on standard error. Text whose only readable parts are punctuation (`Привет, мир.`) speaks 0.46 s with that warning; `Привет мир` with no punctuation is the error above. |
| Most symbols and emoji (`#`, `~`, `/`, `😀`) | Left out with no warning. Some symbols are read as words: `&` as "and", `%` as "percent", `+` as "plus", `=` as "equals", `@` as "at", `©` as "copyright". |
| A whole number of more than 306 digits | Left out with no warning; a 307-digit number leaves only its full stop. |
| Length | A text of 5,016 characters (96 sentences) speaks in full: 306 s of speech in 96 s, peak memory about 307 MB. A text of 50,000 characters (8,908 words) also speaks in full: 3,049 s of speech in 961 s. One sentence of 11,263 characters with no full stop speaks in full in the Python tests (644 s of speech). The text is split into pieces of at most 510 phonemes; a longer sentence is split too, with no warning. |
| Unknown voice ID | `Err`: `no voice named 'labzero-ai/unknown'; the voices are: labzero-ai/zero-talk-11m` |
| `Voice::from_files` with the files missing | `Err`: `<folder>/frontend.bin: No such file or directory` (the front-end file is opened first) |
| `Voice::from_files` with a file of another kind | `Err`: `<folder>/model.bin: this is a 'frontend' file, not a model file` |
| `save()` with another extension (`.mp3`, none) | `Err`: `<path>: save() writes .wav (WAV) or .ogg (Ogg Opus); name the file with one of those extensions`. Nothing is written. `.WAV` in capitals works. |
| `save()` into a missing folder | `Err`: `<path>: No such file or directory` |
| Command options | `zero-talk: --bitrate needs a whole number of kbit/s from 8 to 128, not '999'`; `zero-talk: x.mp3: the output file must end in .ogg (Ogg Opus) or .wav (WAV); MP3 is not built`; `--format` takes `ogg`, `wav` or `pcm`. |

Streaming: `stream()` yields one `Result<Chunk>` for each piece as it is made; for a three-sentence text that is 11 chunks,
the first of 55 samples (about 2 ms), and the chunks joined equal the samples of `speak()`.

## Licence

Apache-2.0, for the code and the model files. Zero Talk builds on Kokoro-82M (Apache-2.0). Third-party notices are in
[THIRD-PARTY-NOTICES.md](https://github.com/labzero-ai/zero-talk/blob/main/rust/THIRD-PARTY-NOTICES.md).

Lab Zero · https://labzero.ai
