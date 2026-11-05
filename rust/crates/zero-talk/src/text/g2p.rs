//! misaki 0.9.4's G2P.retokenize / resolve_tokens / merge_tokens / __call__ for plain text (no inline cues), the word lookup being
//! the Lexicon. Tokens come from the analyzer as spaCy-style (text, tag, whitespace) triples.
//! Ported from misaki 0.9.4 (Apache-2.0) and changed for this engine.
use super::lexicon::*;
use super::pyutil::*;

const SUBTOKEN_JUNKS: &str = "',-._‘’/";
const PUNCTS: &str = ";:,.!?—…\"“”";
const PUNCT_TAGS: [&str; 11] = [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"];
const QUOTES: &str = "'‘’";

fn punct_tag_phonemes(tag: &str) -> Option<&'static str> {
    match tag {
        "-LRB-" => Some("("),
        "-RRB-" => Some(")"),
        "``" => Some("“"),
        "\"\"" | "''" => Some("”"),
        _ => None,
    }
}

fn is_nd(c: char) -> bool {
    c.is_ascii_digit() || decimal_value(c).is_some()
}

// misaki's SUBTOKEN_REGEX, by alternatives in order:
//   ^['‘’]+ | \p{Lu}(?=\p{Lu}\p{Ll}) | (?:^-)?(?:\p{Nd}?[,.]?\p{Nd})+ | [-_]+ | ['‘’]{2,} |
//   \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu}) | \p{L}+(?:['‘’]\p{L})* | [^-_\p{L}'‘’\p{Nd}] | ['‘’]+$
fn digit_unit(c: &[char], i: usize) -> Option<usize> {
    let at = |k: usize| c.get(k).copied();
    for take_d in [true, false] {
        for take_s in [true, false] {
            let mut k = i;
            if take_d {
                if !at(k).is_some_and(is_nd) {
                    continue;
                }
                k += 1;
            }
            if take_s {
                if !at(k).is_some_and(|x| x == ',' || x == '.') {
                    continue;
                }
                k += 1;
            }
            if at(k).is_some_and(is_nd) {
                return Some(k + 1);
            }
        }
    }
    None
}

fn match_at(c: &[char], i: usize) -> Option<usize> {
    let n = c.len();
    let at = |k: usize| c.get(k).copied();
    let is_q = |x: char| QUOTES.contains(x);
    // 1
    if i == 0 && at(0).is_some_and(is_q) {
        return Some(c.iter().take_while(|&&x| is_q(x)).count());
    }
    // 2
    if at(i).is_some_and(|x| x.is_uppercase() && is_letter(x)) && at(i + 1).is_some_and(|x| x.is_uppercase() && is_letter(x)) && at(i + 2).is_some_and(|x| x.is_lowercase() && is_letter(x)) {
        return Some(i + 1);
    }
    // 3
    {
        let mut p = i;
        if i == 0 && at(0) == Some('-') {
            p = 1;
        }
        let mut end = None;
        let mut q = p;
        while let Some(e) = digit_unit(c, q) {
            q = e;
            end = Some(e);
        }
        if end.is_some() {
            return end;
        }
    }
    // 4
    if at(i).is_some_and(|x| x == '-' || x == '_') {
        return Some(i + c[i..].iter().take_while(|&&x| x == '-' || x == '_').count());
    }
    // 5
    if at(i).is_some_and(is_q) && at(i + 1).is_some_and(is_q) {
        return Some(i + c[i..].iter().take_while(|&&x| is_q(x)).count());
    }
    // 6: lazy letters, lazy (quote letter) groups, then a lower-case letter before an upper-case one
    {
        let mut k = i;
        loop {
            // k letters consumed so far (lazy: increase after the groups are tried)
            let mut p = k;
            loop {
                if at(p).is_some_and(|x| x.is_lowercase() && is_letter(x)) && at(p + 1).is_some_and(|x| x.is_uppercase() && is_letter(x)) {
                    return Some(p + 1);
                }
                if at(p).is_some_and(is_q) && at(p + 1).is_some_and(is_letter) {
                    p += 2;
                } else {
                    break;
                }
            }
            if at(k).is_some_and(is_letter) {
                k += 1;
            } else {
                break;
            }
        }
    }
    // 7
    if at(i).is_some_and(is_letter) {
        let mut p = i;
        while at(p).is_some_and(is_letter) {
            p += 1;
        }
        while at(p).is_some_and(is_q) && at(p + 1).is_some_and(is_letter) {
            p += 2;
        }
        return Some(p);
    }
    // 8
    if at(i).is_some_and(|x| !(x == '-' || x == '_' || is_letter(x) || is_q(x) || is_nd(x))) {
        return Some(i + 1);
    }
    // 9
    if at(i).is_some_and(is_q) && c[i..].iter().all(|&x| is_q(x)) && i < n {
        return Some(n);
    }
    None
}

pub fn subtokenize(word: &str) -> Vec<String> {
    let c: Vec<char> = word.chars().collect();
    let (mut out, mut i) = (Vec::new(), 0);
    while i < c.len() {
        match match_at(&c, i) {
            Some(e) if e > i => {
                out.push(c[i..e].iter().collect());
                i = e;
            }
            _ => i += 1,
        }
    }
    out
}

pub fn mk_token(text: &str, tag: &str, whitespace: &str) -> Token {
    Token { text: text.to_string(), tag: tag.to_string(), whitespace: whitespace.to_string(), phonemes: None, u: Under::new() }
}

fn stress_weight(ps: &str) -> usize {
    ps.chars().map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 }).sum()
}

pub fn merge_tokens(tokens: &[Token], unk: Option<&str>) -> Token {
    let mut stresses: Vec<f64> = Vec::new();
    for t in tokens {
        if let Some(s) = t.u.stress {
            if !stresses.contains(&s) {
                stresses.push(s);
            }
        }
    }
    let currency = tokens.iter().filter_map(|t| t.u.currency).max();
    let phonemes = unk.map(|unk| {
        let mut ps = String::new();
        for tk in tokens {
            let last = ps.chars().next_back();
            if tk.u.prespace && !ps.is_empty() && !last.is_some_and(is_space) && tk.phonemes.as_deref().is_some_and(|p| !p.is_empty()) {
                ps.push(' ');
            }
            ps.push_str(tk.phonemes.as_deref().unwrap_or(unk));
        }
        ps
    });
    let (mut best, mut best_w) = (&tokens[0], 0usize);
    let mut first = true;
    for tk in tokens {
        let w: usize = tk.text.chars().map(|c| if c.to_lowercase().eq(std::iter::once(c)) { 1 } else { 2 }).sum();
        if first || w > best_w {
            best = tk;
            best_w = w;
            first = false;
        }
    }
    let mut text = String::new();
    for tk in &tokens[..tokens.len() - 1] {
        text.push_str(&tk.text);
        text.push_str(&tk.whitespace);
    }
    text.push_str(&tokens[tokens.len() - 1].text);
    let mut flags: Vec<char> = tokens.iter().flat_map(|t| t.u.num_flags.chars()).collect();
    flags.sort_unstable();
    flags.dedup();
    Token {
        text,
        tag: best.tag.clone(),
        whitespace: tokens[tokens.len() - 1].whitespace.clone(),
        phonemes,
        u: Under { is_head: tokens[0].u.is_head, alias: None, stress: if stresses.len() == 1 { Some(stresses[0]) } else { None }, currency, num_flags: flags.into_iter().collect(), prespace: tokens[0].u.prespace },
    }
}

enum Word {
    One(Token),
    Group(Vec<Token>),
}

fn retokenize(tokens: &[Token]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut currency: Option<char> = None;
    for (i, token) in tokens.iter().enumerate() {
        let mut tks: Vec<Token> = if token.u.alias.is_none() && token.phonemes.is_none() {
            subtokenize(&token.text)
                .into_iter()
                .map(|t| Token { text: t, tag: token.tag.clone(), whitespace: String::new(), phonemes: None, u: Under { is_head: true, num_flags: token.u.num_flags.clone(), stress: token.u.stress, ..Under::new() } })
                .collect()
        } else {
            vec![token.clone()]
        };
        if tks.is_empty() {
            continue;
        }
        tks.last_mut().unwrap().whitespace = token.whitespace.clone();
        for j in 0..tks.len() {
            let mut tk = tks[j].clone();
            let (has_alias, has_ps) = (tk.u.alias.is_some(), tk.phonemes.is_some());
            if has_alias || has_ps {
            } else if tk.tag == "$" && matches!(tk.text.as_str(), "$" | "£" | "€") {
                currency = tk.text.chars().next();
                tk.phonemes = Some(String::new());
            } else if tk.tag == ":" && (tk.text == "-" || tk.text == "–") {
                tk.phonemes = Some("—".to_string());
            } else if PUNCT_TAGS.contains(&tk.tag.as_str()) && !tk.text.chars().all(|c| c.to_lowercase().next().is_some_and(|l| ('a'..='z').contains(&l))) {
                tk.phonemes = Some(punct_tag_phonemes(&tk.tag).map_or_else(|| tk.text.chars().filter(|&c| PUNCTS.contains(c)).collect(), str::to_string));
            } else if currency.is_some() {
                if tk.tag != "CD" {
                    currency = None;
                } else if j + 1 == tks.len() && (i + 1 == tokens.len() || tokens[i + 1].tag != "CD") {
                    tk.u.currency = currency;
                }
            } else if j > 0 && j + 1 < tks.len() && tk.text == "2" && isalpha(&format!("{}{}", last_char(&tks[j - 1].text).unwrap_or(' '), first_char(&tks[j + 1].text))) {
                tk.u.alias = Some("to".to_string());
            }
            if tk.u.alias.is_some() || tk.phonemes.is_some() {
                words.push(Word::One(tk));
            } else if matches!(words.last(), Some(Word::Group(g)) if g.last().is_some_and(|t| t.whitespace.is_empty())) {
                tk.u.is_head = false;
                if let Some(Word::Group(g)) = words.last_mut() {
                    g.push(tk);
                }
            } else if !tk.whitespace.is_empty() {
                words.push(Word::One(tk));
            } else {
                words.push(Word::Group(vec![tk]));
            }
        }
    }
    words.into_iter().map(|w| match w {
        Word::Group(mut g) if g.len() == 1 => Word::One(g.pop().unwrap()),
        w => w,
    }).collect()
}

fn token_context(ctx: Ctx, ps: Option<&str>, token: &Token) -> Ctx {
    let mut vowel = ctx.future_vowel;
    if let Some(ps) = ps {
        for c in ps.chars() {
            if VOWELS.contains(c) || CONSONANTS.contains(c) || NON_QUOTE_PUNCTS.contains(c) {
                vowel = if NON_QUOTE_PUNCTS.contains(c) { None } else { Some(VOWELS.contains(c)) };
                break;
            }
        }
    }
    let t = token.text.as_str();
    Ctx { future_vowel: vowel, future_to: t == "to" || t == "To" || (t == "TO" && (token.tag == "TO" || token.tag == "IN")) }
}

fn all_junk(s: &str) -> bool {
    s.chars().all(|c| SUBTOKEN_JUNKS.contains(c))
}

fn resolve_tokens(tokens: &mut [Token]) {
    let mut text = String::new();
    for tk in &tokens[..tokens.len() - 1] {
        text.push_str(&tk.text);
        text.push_str(&tk.whitespace);
    }
    text.push_str(&tokens[tokens.len() - 1].text);
    let kinds: std::collections::HashSet<u8> = text.chars().filter(|&c| !SUBTOKEN_JUNKS.contains(c)).map(|c| if is_letter(c) { 0 } else if c.is_ascii_digit() { 1 } else { 2 }).collect();
    let prespace = text.contains(' ') || text.contains('/') || kinds.len() > 1;
    let last = tokens.len() - 1;
    for (i, tk) in tokens.iter_mut().enumerate() {
        if tk.phonemes.is_none() {
            if i == last && tk.text.chars().count() == 1 && NON_QUOTE_PUNCTS.contains(tk.text.as_str()) {
                tk.phonemes = Some(tk.text.clone());
            } else if all_junk(&tk.text) {
                tk.phonemes = Some(String::new());
            }
        } else if i > 0 {
            tk.u.prespace = prespace;
        }
    }
    if prespace {
        return;
    }
    let mut indices: Vec<(usize, usize, usize)> = Vec::new();
    for (i, tk) in tokens.iter().enumerate() {
        if let Some(ps) = tk.phonemes.as_deref().filter(|p| !p.is_empty()) {
            indices.push((usize::from(ps.contains(PRIMARY)), stress_weight(ps), i));
        }
    }
    if indices.len() == 2 && cp_len(&tokens[indices[0].2].text) == 1 {
        let i = indices[1].2;
        tokens[i].phonemes = apply_stress(tokens[i].phonemes.as_deref(), Some(-0.5));
        return;
    }
    if indices.len() < 2 || indices.iter().map(|x| x.0).sum::<usize>() <= (indices.len() + 1) / 2 {
        return;
    }
    indices.sort();
    for &(_, _, i) in indices.iter().take(indices.len() / 2) {
        tokens[i].phonemes = apply_stress(tokens[i].phonemes.as_deref(), Some(-0.5));
    }
}

/// G2P.__call__ on spaCy tokens (text, tag, whitespace): the final tokens with their phonemes.
pub fn g2p(lexicon: &Lexicon, spacy: &[(String, String, String)], unk: &str) -> Vec<Token> {
    let tokens: Vec<Token> = spacy.iter().map(|(t, g, w)| mk_token(t, g, w)).collect();
    let mut words = retokenize(&tokens);
    let mut ctx = Ctx { future_vowel: None, future_to: false };
    for i in (0..words.len()).rev() {
        match &mut words[i] {
            Word::One(w) => {
                if w.phonemes.is_none() {
                    w.phonemes = lexicon.call(w, &ctx);
                }
                ctx = token_context(ctx, w.phonemes.as_deref(), w);
            }
            Word::Group(w) => {
                let (mut left, mut right) = (0usize, w.len());
                while left < right {
                    let part = &w[left..right];
                    let tk = if !part.iter().any(|x| x.u.alias.is_some() || x.phonemes.is_some()) { Some(merge_tokens(part, None)) } else { None };
                    let ps = tk.as_ref().and_then(|t| lexicon.call(t, &ctx));
                    if let Some(ps) = ps {
                        ctx = token_context(ctx, Some(&ps), tk.as_ref().unwrap());
                        w[left].phonemes = Some(ps);
                        for x in &mut w[left + 1..right] {
                            x.phonemes = Some(String::new());
                        }
                        right = left;
                        left = 0;
                    } else if left + 1 < right {
                        left += 1;
                    } else {
                        right -= 1;
                        if w[right].phonemes.is_none() && all_junk(&w[right].text) {
                            w[right].phonemes = Some(String::new());
                        }
                        left = 0;
                    }
                }
                resolve_tokens(w);
            }
        }
    }
    words
        .into_iter()
        .map(|w| match w {
            Word::One(t) => t,
            Word::Group(g) => merge_tokens(&g, Some(unk)),
        })
        .map(|mut tk| {
            if let Some(p) = tk.phonemes.as_mut().filter(|p| !p.is_empty()) {
                *p = p.replace('ɾ', "T").replace('ʔ', "t");
            }
            tk
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::subtokenize;

    #[test]
    fn subtokens_of_misakis_pattern() {
        let t = |s: &str| subtokenize(s);
        assert_eq!(t("O'Brien"), ["O'B", "rien"]);
        assert_eq!(t("don't"), ["don't"]);
        assert_eq!(t("McDonald"), ["Mc", "Donald"]);
        assert_eq!(t("1,000.50"), ["1,000.50"]);
        assert_eq!(t("e-mail"), ["e", "-", "mail"]);
        assert_eq!(t("DAILYo'farrell"), ["DAILYo'f", "arrell"]);
        assert_eq!(t("Philosopher'sconcerningFOREVERcheerful"), ["Philosopher's", "concerning", "FOREVERcheerful"]);
        assert_eq!(t("1stejaculatedshe'dTrying"), ["1", "stejaculatedshe'd", "Trying"]);
        assert_eq!(t("~BENNYDECK'Spiled"), ["~", "BENNYDECK'S", "piled"]);
    }
}
