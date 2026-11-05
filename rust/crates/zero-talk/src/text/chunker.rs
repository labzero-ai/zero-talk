//! The chunking of English text: tokens become phoneme chunks of at most 510 phonemes, after kokoro 0.9.4's pipeline
//! (tokens_to_ps, waterfall_last, en_tokenize), and the grouping into sentences (`sentences`).
//! Ported from kokoro 0.9.4 (Apache-2.0) and changed for this engine.
use super::lexicon::Token;
use super::pyutil::{cp_len, lstrip, rstrip, strip};

pub fn tokens_to_ps(tokens: &[Token]) -> String {
    let s: String = tokens.iter().map(|t| format!("{}{}", t.phonemes.as_deref().unwrap_or(""), if t.whitespace.is_empty() { "" } else { " " })).collect();
    strip(&s).to_string()
}

const WATERFALL: [&str; 3] = ["!.?…", ":;", ",—"];
const BUMPS: [&str; 2] = [")", "”"];

fn waterfall_last(tokens: &[Token], next_count: usize) -> usize {
    for w in WATERFALL {
        let Some(z) = tokens.iter().rposition(|t| t.phonemes.as_deref().is_some_and(|p| cp_len(p) == 1 && w.contains(p))) else { continue };
        let mut z = z + 1;
        if z < tokens.len() && BUMPS.contains(&tokens[z].phonemes.as_deref().unwrap_or("")) {
            z += 1;
        }
        if next_count - cp_len(&tokens_to_ps(&tokens[..z])) <= 510 {
            return z;
        }
    }
    tokens.len()
}

/// KPipeline.en_tokenize: the phonemes of each chunk of at most 510 (cut at sentence-ending punctuation if there is any, else at
/// : ; , —).
pub fn en_tokenize(tokens: Vec<Token>) -> Vec<String> {
    let mut out = Vec::new();
    let (mut tks, mut pcount): (Vec<Token>, usize) = (Vec::new(), 0);
    for mut t in tokens {
        t.phonemes.get_or_insert_with(String::new);
        let mut next_ps = format!("{}{}", t.phonemes.as_deref().unwrap(), if t.whitespace.is_empty() { "" } else { " " });
        let next_pcount = pcount + cp_len(rstrip(&next_ps));
        if next_pcount > 510 {
            let z = waterfall_last(&tks, next_pcount);
            out.push(tokens_to_ps(&tks[..z]));
            tks.drain(..z);
            pcount = cp_len(&tokens_to_ps(&tks));
            if tks.is_empty() {
                next_ps = lstrip(&next_ps).to_string();
            }
        }
        pcount += cp_len(&next_ps);
        tks.push(t);
    }
    if !tks.is_empty() {
        out.push(tokens_to_ps(&tks));
    }
    out
}

/// Sentence grouping: a token made only of sentence-ending marks ends a group.
pub fn sentences(tokens: Vec<Token>) -> Vec<Vec<Token>> {
    let (mut out, mut cur) = (Vec::new(), Vec::new());
    for t in tokens {
        let end = !t.text.is_empty() && t.text.chars().all(|c| matches!(c, '.' | '!' | '?' | '…'));
        cur.push(t);
        if end {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
