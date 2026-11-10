//! Zero Talk 11M: a small English text-to-speech model, and the engine that runs it.
//!
//! The model is built into the crate, so a program speaks on its own: no Python, no GPU and no network access.
//!
//! ```no_run
//! use zero_talk::Voice;
//!
//! let voice = Voice::load("labzero-ai/zero-talk-11m")?;
//! let audio = voice.speak("Hello from a small voice.")?;
//!
//! audio.save("hello.wav")?; // the extension picks the format: .wav or .ogg (Ogg Opus)
//! let samples: &[f32] = audio.samples(); // mono, 24 kHz
//!
//! for chunk in voice.stream("A longer text, read piece by piece.")? {
//!     let chunk = chunk?;
//!     let samples: &[f32] = chunk.samples();
//!     println!("{} samples", samples.len());
//! }
//! # Ok::<(), zero_talk::Error>(())
//! ```
//!
//! Start with [`Voice`]. The text is English: plain sentences, with numbers, dates and prices written as digits read out in words.
#![warn(missing_docs)]

// With the `internals` feature the engine's modules are public, for the WebAssembly build (crates/zero-talk-wasm), which drives the
// stages of the pipeline itself. They are not part of the crate's API.
macro_rules! modules {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg(feature = "internals")]
            #[doc(hidden)]
            #[allow(missing_docs)]
            pub mod $name;
            #[cfg(not(feature = "internals"))]
            mod $name;
        )*
    };
}

modules!(archive, audio, conditioner, error, generator, kernels, marker, opus, pipeline, plan, post, source, speak, stream, student, text, voice, wav);

mod play;

#[doc(hidden)]
pub mod cli;

pub use audio::{Audio, Format};
pub use error::{Error, Result};
pub use stream::{Chunk, Stream};
pub use voice::{voices, Voice};
