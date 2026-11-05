//! The files of Zero Talk 11M: the model (`model.bin`) and the text front end (`frontend.bin`), built into this crate as bytes.
//!
//! The `zero-talk` crate reads them through `Voice::load("labzero-ai/zero-talk-11m")`, so a program needs no file on disk. The
//! files are the release files of the model: their sizes and SHA-256 hashes are the ones listed in the model's `model.json`.
//! The model is licensed under Apache-2.0 (see `LICENSE`, `NOTICE` and `THIRD-PARTY-NOTICES.md`).

#![warn(missing_docs)]

/// The model file: the network weights, the voice table and the model's settings.
pub const MODEL: &[u8] = include_bytes!("../data/model.bin");

/// The text front-end file: the pronunciation tables, the tokenizer and tagger data, and the G2P network.
pub const FRONTEND: &[u8] = include_bytes!("../data/frontend.bin");

/// The size of [`MODEL`] in bytes.
pub const MODEL_BYTES: usize = 7_053_248;

/// The SHA-256 hash of [`MODEL`], in lowercase hexadecimal.
pub const MODEL_SHA256: &str = "f0e885226e5fab84d9a800926479e420fc19161a5b4b435eb39d4ee102431ac5";

/// The size of [`FRONTEND`] in bytes.
pub const FRONTEND_BYTES: usize = 816_248;

/// The SHA-256 hash of [`FRONTEND`], in lowercase hexadecimal.
pub const FRONTEND_SHA256: &str = "5ade509c060811a5fe9525ac554d6edc2229c2fca66581e093463b9d9b8eaffd";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_files_have_the_listed_sizes() {
        assert_eq!(MODEL.len(), MODEL_BYTES);
        assert_eq!(FRONTEND.len(), FRONTEND_BYTES);
    }

    #[test]
    fn the_files_are_containers_of_zero_talk() {
        assert_eq!(&MODEL[..4], b"ZERO");
        assert_eq!(&FRONTEND[..4], b"ZERO");
    }
}
