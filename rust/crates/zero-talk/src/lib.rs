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

mod archive;
mod audio;
mod conditioner;
mod error;
mod generator;
mod kernels;
mod marker;
mod opus;
mod pipeline;
mod plan;
mod play;
mod post;
mod source;
mod speak;
mod stream;
mod student;
mod text;
mod voice;
mod wav;

#[doc(hidden)]
pub mod cli;

pub use audio::Audio;
pub use error::{Error, Result};
pub use stream::{Chunk, Stream};
pub use voice::{voices, Voice};
