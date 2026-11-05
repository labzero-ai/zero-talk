//! The text front end: text to sentence chunks of phonemes. Its parts are a port of spaCy's tokenizer, a rule-based tagger,
//! misaki's English rules, kokoro's chunking, and a small G2P network with stored tables for the words the lexicon does not know.
//! Ported from misaki 0.9.4 and kokoro 0.9.4 (Apache-2.0) and spaCy 3.8 (MIT), and changed for this engine.
pub mod chunker;
pub mod fold;
pub mod frontend;
pub mod g2p;
pub mod lexicon;
pub mod network;
pub mod num2words;
pub mod pyutil;
pub mod symbols;
pub mod tables;
pub mod tokenizer;
pub mod tagger;
