//! The AI-generated marker: metadata that says the audio is AI-generated. The WAV and Ogg Opus writers take their text from here, so
//! both formats carry the same marker. Raw PCM has no metadata and carries none.

/// The comment of a file: the WAV's `ICMT` item and the Ogg Opus `COMMENT` tag.
pub const COMMENT: &str = "AI-generated speech from Zero Talk 11M (labzero-ai/zero-talk-11m)";

/// The software that made a file: the WAV's `ISFT` item.
pub const SOFTWARE: &str = "Zero Talk 11M";

/// The IPTC digital source type of AI-generated media: the Ogg Opus `DIGITALSOURCETYPE` tag.
pub const SOURCE: &str = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia";
