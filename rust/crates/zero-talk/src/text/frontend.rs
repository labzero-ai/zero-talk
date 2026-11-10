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
    /// Steering tags (`[M2]`, `[breathy:1]`) that were removed from the text, in order. Zero Talk 11M has no steering.
    pub tags: Vec<String>,
}

/// Removes mode and cue tags, `[M0]`...`[M9]` and `[word:number]` such as `[breathy:1]`. This model has no steering, so they
/// must not be read aloud. A link label `[label](target)` is not a tag. Returns the text and the tags removed.
pub fn strip_tags(text: &str) -> (String, Vec<String>) {
    let t: Vec<char> = text.chars().collect();
    let (mut out, mut tags) = (String::new(), Vec::new());
    let mut i = 0;
    while i < t.len() {
        if t[i] == '[' {
            if let Some(close) = (i + 1..t.len()).find(|&k| t[k] == ']') {
                let body: String = t[i + 1..close].iter().collect();
                if is_tag(&body) {
                    tags.push(format!("[{body}]"));
                    i = close + 1;
                    // "a [M2] b" loses the tag and one of the two spaces around it.
                    if t.get(i) == Some(&' ') && (out.is_empty() || out.ends_with(char::is_whitespace)) {
                        i += 1;
                    }
                    continue;
                }
            }
        }
        out.push(t[i]);
        i += 1;
    }
    (out, tags)
}

fn is_tag(body: &str) -> bool {
    let mode = body.len() >= 2 && body.starts_with(['M', 'm']) && body[1..].chars().all(|c| c.is_ascii_digit());
    let cue = body.split_once(':').is_some_and(|(name, value)| {
        !name.is_empty() && name.chars().all(|c| c.is_ascii_alphabetic() || c == '_') && !value.is_empty() && value.chars().all(|c| c.is_ascii_digit() || c == '.')
    });
    mode || cue
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
        let (text, tags) = strip_tags(text);
        let mut plan = Plan { chunks: Vec::new(), unread: Vec::new(), tags };
        for segment in strip(&text).split('\n').filter(|s| !strip(s).is_empty()) {
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

#[cfg(test)]
mod tests {
    use super::strip_tags;

    fn s(t: &str) -> (String, Vec<String>) {
        strip_tags(t)
    }

    #[test]
    fn mode_and_cue_tags_are_removed_and_named() {
        assert_eq!(s("[M2] Hi."), ("Hi.".to_string(), vec!["[M2]".to_string()]));
        assert_eq!(s("[breathy:1] Hi."), ("Hi.".to_string(), vec!["[breathy:1]".to_string()]));
        assert_eq!(s("Hi [M0] there."), ("Hi there.".to_string(), vec!["[M0]".to_string()]));
        assert_eq!(s("a[M4]b").0, "ab");
        assert_eq!(s("[M1] one. [M3] two.").1.len(), 2);
    }

    #[test]
    fn other_brackets_are_left_alone() {
        for t in ["No tags here.", "See [the docs](http://x.io) now.", "An [aside] here.", "Array [1] and [a:b].", "[M] alone", "Open [bracket"] {
            assert_eq!(s(t), (t.to_string(), vec![]), "{t}");
        }
    }
}
