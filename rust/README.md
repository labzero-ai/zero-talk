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

Text: English, plain sentences; numbers, dates and prices written as digits are read out. Other scripts are skipped with a
warning on `audio.warnings()`.

## Where the model files come from

`zero-talk-model-11m` holds the model and front-end files, the same bytes as the model page lists (sizes and SHA-256 in
`model.json`). The engine embeds them through its default `embed` feature, and `Voice::load` finds them by ID. Keeping
the data in its own crate means an engine update does not re-publish the model. To ship your own copy instead:

```toml
zero-talk = { version = "1", default-features = false }
```

## What it does and does not do

English only, one voice, 24 kHz mono. Questions come out flat, and long sentences can lose their rhythm; short, plain
sentences work best. The full list is on the model page.

The tests pass on macOS (Apple silicon), Linux (x86-64 and ARM) and Windows (x86-64). The audio is the same on every
run on one system; it differs between operating systems, system versions and processors by floating-point rounding.

## Licence

Apache-2.0, for the code and the model files. Zero Talk builds on Kokoro-82M (Apache-2.0). Third-party notices are in
[THIRD-PARTY-NOTICES.md](https://github.com/labzero-ai/zero-talk/blob/main/rust/THIRD-PARTY-NOTICES.md).

Lab Zero · https://labzero.ai
