//! The speech that [`Voice::speak`](crate::Voice::speak) makes, and the files it can be written to.
use std::path::Path;

use crate::archive::io_message;
use crate::error::{Error, Result};
use crate::opus;
use crate::pipeline::SAMPLE_RATE;
use crate::wav;

/// A file format for [`Audio::encode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// 16-bit PCM WAV, mono, 24 kHz.
    Wav,
    /// Ogg Opus, mono, 24 kHz, at `bitrate` kbit/s (from 8 to 128; [`Format::OGG`] uses 24).
    Ogg {
        /// The bit rate in kbit/s.
        bitrate: u32,
    },
}

impl Format {
    /// Ogg Opus at the default bit rate, 24 kbit/s.
    pub const OGG: Format = Format::Ogg { bitrate: opus::DEFAULT_KBPS };
}

/// Speech as mono samples at 24 kHz, from [`Voice::speak`](crate::Voice::speak).
///
/// ```no_run
/// # fn main() -> zero_talk::Result<()> {
/// let voice = zero_talk::Voice::load("labzero-ai/zero-talk-11m")?;
/// let audio = voice.speak("Hello from a small voice.")?;
/// audio.save("hello.ogg")?; // the extension picks the format: .ogg (Ogg Opus) or .wav
/// let bytes = audio.encode(zero_talk::Format::Ogg { bitrate: 64 })?; // the file's bytes, in memory
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    samples: Vec<f32>,
    warnings: Vec<String>,
}

impl Audio {
    pub(crate) fn new(samples: Vec<f32>, warnings: Vec<String>) -> Self {
        Audio { samples, warnings }
    }

    /// The samples: mono, from -1 to 1, at [`sample_rate`](Self::sample_rate).
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// The sample rate of [`samples`](Self::samples), in hertz: always 24000.
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// The length of the speech in seconds.
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / SAMPLE_RATE as f64
    }

    /// What was left out of the text, one message per problem: words that could not be read, and characters that the model has
    /// no sound for. Empty when everything was spoken.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Writes the speech to `path`. The extension picks the format: `.wav` (16-bit PCM WAV) or `.ogg` (Ogg Opus, 24 kbit/s).
    /// Both carry an AI-generated marker in their metadata. Any other extension is an error.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
            Some("wav") => self.encode(Format::Wav)?,
            Some("ogg") => self.encode(Format::OGG)?,
            _ => {
                return Err(Error::new(format!(
                    "{}: save() writes .wav (WAV) or .ogg (Ogg Opus); name the file with one of those extensions",
                    path.display()
                )))
            }
        };
        std::fs::write(path, bytes).map_err(|e| {
            let hint = if e.kind() == std::io::ErrorKind::NotFound { "; create the folder first" } else { "" };
            Error::new(format!("{}: {}{}", path.display(), io_message(&e), hint))
        })
    }

    /// The speech as a complete file in memory, in `format`: [`Format::Wav`] or [`Format::Ogg`] (see [`Format::OGG`] for the
    /// default bit rate). Both carry an AI-generated marker in their metadata. An Ogg bit rate outside 8 to 128 kbit/s is an error.
    pub fn encode(&self, format: Format) -> Result<Vec<u8>> {
        match format {
            Format::Wav => Ok(wav::encode(&self.samples, SAMPLE_RATE)),
            Format::Ogg { bitrate } => {
                if !opus::KBPS.contains(&bitrate) {
                    return Err(Error::new(format!(
                        "bitrate must be a whole number of kbit/s from {} to {}, not {bitrate}",
                        opus::KBPS.start(),
                        opus::KBPS.end()
                    )));
                }
                opus::encode_ogg(&self.samples, SAMPLE_RATE, bitrate).map_err(Error::from)
            }
        }
    }
}
