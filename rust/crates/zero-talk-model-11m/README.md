# Zero Talk 11M model files

The model and English text front-end files of [Zero Talk 11M](https://labzero.ai/zero-talk-11m/), compiled into a Rust
crate as byte slices. Most programs want the [zero-talk crate](https://crates.io/crates/zero-talk) instead: it loads the
voice, speaks text and streams audio, and it embeds these files by default. Depend on this crate directly only to read or
copy the files.

```rust
zero_talk_model_11m::MODEL      // &[u8], the model (7,053,248 bytes)
zero_talk_model_11m::FRONTEND   // &[u8], the text front end (816,248 bytes)
// with their sizes and SHA-256 hashes:
// MODEL_BYTES, MODEL_SHA256, FRONTEND_BYTES, FRONTEND_SHA256
```

The files only work with the `zero-talk` engine; they do not load in PyTorch or `transformers`.

Source and issues: [labzero-ai/zero-talk](https://github.com/labzero-ai/zero-talk) ·
[issues](https://github.com/labzero-ai/zero-talk/issues).

Apache-2.0. See [LICENSE](https://github.com/labzero-ai/zero-talk/blob/main/rust/crates/zero-talk-model-11m/LICENSE),
[NOTICE](https://github.com/labzero-ai/zero-talk/blob/main/rust/crates/zero-talk-model-11m/NOTICE) and
[third-party notices](https://github.com/labzero-ai/zero-talk/blob/main/rust/crates/zero-talk-model-11m/THIRD-PARTY-NOTICES.md).
