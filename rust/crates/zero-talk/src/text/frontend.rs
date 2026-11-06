//! The text front end: text to sentence chunks of phonemes, each with its voice-table row.
//! Ported from misaki 0.9.4 and kokoro 0.9.4 (Apache-2.0) and changed for this engine.
use super::chunker::{en_tokenize, sentences};
use super::g2p::g2p;
use super::lexicon::{Lexicon, Token};
use super::network::G2pNet;
use super::pyutil::{is_letter, lstrip, strip};
use super::tables::Tables;
use super::tagger::Tagger;
use super::tokenizer::{st_readings, Tokenizer};
use crate::archive::{Archive, Result};

/// One sentence (or part of a long one) to speak: its phonemes, at most 510, and the voice-table row of that length.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub phonemes: String,
    pub voice_row: usize,
}

/// What the front end makes of a text.
pub struct Plan {
    pub chunks: Vec<Chunk>,
    /// Words that got no phonemes and are left out of the speech, in order of first appearance.
    pub unread: Vec<String>,
}

/// spaCy-style tokens: (text, tag, trailing whitespace).
pub type Spacy = Vec<(String, String, String)>;

pub struct Frontend {
    tables: Tables,
    net: G2pNet,
    tokenizer: Tokenizer,
    tagger: Tagger,
}

/// G2P.preprocess for plain text: leading white space removed, link labels kept (`[label](target)` becomes `label`).
pub fn preprocess(text: &str) -> String {
    let t: Vec<char> = lstrip(text).chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < t.len() {
        if t[i] == '[' {
            if let Some(close) = (i + 1..t.len()).find(|&k| t[k] == ']') {
                if close > i + 1 && t.get(close + 1) == Some(&'(') {
                    if let Some(end) = (close + 2..t.len()).find(|&k| t[k] == ')') {
                        out.extend(&t[i + 1..close]);
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        out.push(t[i]);
        i += 1;
    }
    out
}

impl Frontend {
    pub fn load(ar: &Archive) -> Result<Frontend> {
        let text = |name: &str| ar.bytes(name).and_then(|b| std::str::from_utf8(b).map_err(|e| format!("{name}: {e}")));
        Ok(Frontend { tables: Tables::parse(text("tables")?)?, net: G2pNet::load(ar)?, tokenizer: Tokenizer::load(text("tokenizer")?), tagger: Tagger::load(text("tagger")?)? })
    }

    /// The analysis of one segment: spaCy-style tokens and tags from the tokenizer and the tagger.
    pub fn analyze(&self, text: &str) -> Spacy {
        let toks = st_readings(self.tokenizer.tokenize(text));
        let sentence_upper = text.to_uppercase() == text && text.chars().any(super::pyutil::is_letter);
        let tags = self.tagger.tag(&toks, sentence_upper);
        toks.into_iter().zip(tags).map(|((t, w), g)| (t, g, w)).collect()
    }

    pub fn lexicon(&self) -> Lexicon<'_> {
        Lexicon::new(&self.tables, &self.net)
    }

    /// The final tokens for one line of text, with the given analysis.
    pub fn segment_tokens(&self, lex: &Lexicon, segment: &str, analyze: &mut dyn FnMut(&str) -> Spacy) -> Vec<Token> {
        lex.set_sentence(segment);
        let text = preprocess(segment);
        g2p(lex, &analyze(&text), "")
    }

    /// The chunks of the final tokens of one line: sentence groups, each cut at 510 phonemes.
    pub fn chunks(tokens: Vec<Token>) -> Vec<Chunk> {
        let mut out = Vec::new();
        for group in sentences(tokens) {
            for ps in en_tokenize(group) {
                if ps.is_empty() {
                    continue;
                }
                let ps: String = ps.chars().take(510).collect();
                let voice_row = (ps.chars().count() - 1).min(509);
                out.push(Chunk { phonemes: ps, voice_row });
            }
        }
        out
    }

    /// Text to chunks: each line (a run of newlines separates lines) is analyzed and spoken on its own. Also names the words that
    /// got no phonemes (a letter outside a-z, mostly) and so are left out.
    pub fn plan_with(&self, text: &str, analyze: &mut dyn FnMut(&str) -> Spacy) -> Plan {
        let lex = self.lexicon();
        let mut plan = Plan { chunks: Vec::new(), unread: Vec::new() };
        for segment in strip(text).split('\n').filter(|s| !strip(s).is_empty()) {
            let normalized = super::normalize::normalize(segment);
            let tokens = self.segment_tokens(&lex, &normalized, analyze);
            for t in &tokens {
                if t.text.chars().any(is_letter) && t.phonemes.as_deref().is_none_or(str::is_empty) && !plan.unread.contains(&t.text) {
                    plan.unread.push(t.text.clone());
                }
            }
            plan.chunks.extend(Self::chunks(tokens));
        }
        plan
    }

    pub fn plan(&self, text: &str) -> Plan {
        self.plan_with(text, &mut |t| self.analyze(t))
    }
}
