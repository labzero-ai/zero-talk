//! misaki 0.9.4's English lexicon (en.py), as ported: special cases, the word cascade, -s/-ed/-ing derivation,
//! numbers/currency/years, acronym spelling and stress. The word lookup is the stored tables, then the network.
//! Ported from misaki 0.9.4 (Apache-2.0) and changed for this engine.
use std::cell::Cell;

use super::fold::fold_word;
use super::network::G2pNet;
use super::num2words::{cardinal, float_words, ordinal, year, Big};
use super::pyutil::*;
use super::tables::{Stored, Tables};

pub const PRIMARY: char = 'ˈ';
pub const SECONDARY: char = 'ˌ';
const STRESSES: &str = "ˌˈ";
pub const DIPHTHONGS: &str = "AIOQWYʤʧ";
pub const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
pub const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
pub const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const ORDINALS: [&str; 4] = ["st", "nd", "rd", "th"];

fn currency_units(c: char) -> Option<(&'static str, &'static str)> {
    match c {
        '$' => Some(("dollar", "cent")),
        '£' => Some(("pound", "pence")),
        '€' => Some(("euro", "cent")),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------------------------- stress
fn has_vowel(ps: &str) -> bool {
    ps.chars().any(|c| VOWELS.contains(c))
}

/// misaki's apply_stress. Values: None, or a number in {-3, -2, -1, -0.5, 0, 0.5, 1, 2}.
pub fn apply_stress(ps: Option<&str>, stress: Option<f64>) -> Option<String> {
    let ps = ps?;
    let Some(stress) = stress else { return Some(ps.to_string()) };
    let has_stress = ps.contains(PRIMARY) || ps.contains(SECONDARY);
    if stress < -1.0 {
        return Some(ps.replace(PRIMARY, "").replace(SECONDARY, ""));
    }
    if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && ps.contains(PRIMARY)) {
        return Some(ps.replace(SECONDARY, "").replace(PRIMARY, &SECONDARY.to_string()));
    }
    if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_stress {
        if !has_vowel(ps) {
            return Some(ps.to_string());
        }
        return Some(restress(&format!("{SECONDARY}{ps}")));
    }
    if stress >= 1.0 && !ps.contains(PRIMARY) && ps.contains(SECONDARY) {
        return Some(ps.replace(SECONDARY, &PRIMARY.to_string()));
    }
    if stress > 1.0 && !has_stress {
        if !has_vowel(ps) {
            return Some(ps.to_string());
        }
        return Some(restress(&format!("{PRIMARY}{ps}")));
    }
    Some(ps.to_string())
}

/// Each stress mark moves to just before the next vowel: key j - 0.5, then a stable sort by key.
fn restress(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    let mut keyed: Vec<(f64, char)> = chars.iter().enumerate().map(|(i, &c)| (i as f64, c)).collect();
    for (i, &p) in chars.iter().enumerate() {
        if STRESSES.contains(p) {
            if let Some(j) = (i..chars.len()).find(|&k| VOWELS.contains(chars[k])) {
                keyed[i] = (j as f64 - 0.5, p);
            }
        }
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    keyed.into_iter().map(|x| x.1).collect()
}

pub fn parent_tag(tag: Option<&str>) -> Option<String> {
    let t = tag?;
    Some(if t.starts_with("VB") {
        "VERB"
    } else if t.starts_with("NN") {
        "NOUN"
    } else if t.starts_with("ADV") || t.starts_with("RB") {
        "ADV"
    } else if t.starts_with("ADJ") || t.starts_with("JJ") {
        "ADJ"
    } else {
        t
    }
    .to_string())
}

// ------------------------------------------------------------------------------------------------ the word predicates
fn is_letters(w: &str) -> bool {
    w.is_ascii() && isalpha(&w.replace('\'', "")) && !w.starts_with('\'') && !w.ends_with('\'')
}
fn shape_ok(w: &str) -> bool {
    w == w.to_lowercase() || w == capitalize(w) || (cp_len(w) > 1 && w == w.to_uppercase())
}
/// Words the network may answer: ASCII letters with inner apostrophes, all lower / Capitalized / UPPER (+ 's).
pub fn eligible(w: &str) -> bool {
    is_letters(w) && (shape_ok(w) || (w.len() > 3 && w.ends_with("'s") && shape_ok(&w[..w.len() - 2])))
}

fn stem_pattern(word: &str) -> bool {
    // /([bcdgklmnprstvxz])\1ing$|cking$/
    if word.ends_with("cking") {
        return true;
    }
    let c: Vec<char> = word.chars().collect();
    let n = c.len();
    n >= 5 && word.ends_with("ing") && c[n - 4] == c[n - 5] && "bcdgklmnprstvxz".contains(c[n - 4])
}

/// What follows the word in the sentence, and whether a "to" is coming (misaki's TokenContext).
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    pub future_vowel: Option<bool>,
    pub future_to: bool,
}

/// A token as misaki's MToken, with its `_` attributes.
#[derive(Clone, Debug)]
pub struct Token {
    pub text: String,
    pub tag: String,
    pub whitespace: String,
    pub phonemes: Option<String>,
    pub u: Under,
}

#[derive(Clone, Debug)]
pub struct Under {
    pub is_head: bool,
    pub alias: Option<String>,
    pub stress: Option<f64>,
    pub currency: Option<char>,
    pub num_flags: String,
    pub prespace: bool,
}

impl Under {
    pub fn new() -> Under {
        Under { is_head: true, alias: None, stress: None, currency: None, num_flags: String::new(), prespace: false }
    }
}

/// Word lookup and the rules around it (the network remembers its answers by word).
pub struct Lexicon<'a> {
    tables: &'a Tables,
    net: &'a G2pNet,
    sentence_upper: Cell<bool>,
    het_only: Cell<bool>,
}

impl<'a> Lexicon<'a> {
    pub fn new(tables: &'a Tables, net: &'a G2pNet) -> Self {
        Lexicon { tables, net, sentence_upper: Cell::new(false), het_only: Cell::new(false) }
    }

    /// Called once per segment before it is processed.
    pub fn set_sentence(&self, text: &str) {
        self.sentence_upper.set(text.to_uppercase() == text && text.chars().any(is_letter));
    }
    fn spell_unknown_caps(&self) -> bool {
        !self.sentence_upper.get()
    }

    fn net_ps(&self, word: &str) -> Option<String> {
        self.net.phonemes(word)
    }

    fn in_golds(&self, w: &str) -> bool {
        self.tables.gold(w).is_some()
    }
    fn in_silvers(&self, w: &str) -> bool {
        self.tables.silver(w).is_some()
    }
    fn gold_plain(&self, w: &str) -> Option<String> {
        match self.tables.gold(w)? {
            Stored::Plain(p) => Some(p.clone()),
            Stored::Het(v) => v.iter().find(|(k, _)| k == "DEFAULT").and_then(|x| x.1.clone()),
        }
    }
    /// The stored entry under exactly `word`, else under its lower case.
    fn stored(&self, silver: bool, word: &str) -> Option<&Stored> {
        let get = |w: &str| if silver { self.tables.silver(w) } else { self.tables.gold(w) };
        let mut v = get(word);
        let lower = word.to_lowercase();
        if v.is_none() && word != lower {
            v = get(&lower);
        }
        v
    }
    fn het_lower(&self, w: &str) -> bool {
        self.tables.gold(w).is_some_and(Stored::is_het) && w == w.to_lowercase()
    }
    fn letter_ps(&self, c: char) -> Option<String> {
        match self.tables.gold(&c.to_uppercase().collect::<String>())? {
            Stored::Plain(p) => Some(p.clone()),
            Stored::Het(_) => None,
        }
    }

    fn get_nnp(&self, word: &str) -> Option<String> {
        let mut ps = String::new();
        for c in word.chars().filter(|&c| is_letter(c)) {
            ps.push_str(&self.letter_ps(c)?);
        }
        let stressed = apply_stress(Some(&ps), Some(0.0))?;
        Some(rsplit1(&stressed, &SECONDARY.to_string()).join(&PRIMARY.to_string()))
    }

    fn special_case(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: &Ctx) -> Option<String> {
        let tag_is = |p: &str| tag.is_some_and(|t| t.starts_with(p));
        if tag == Some("ADD") && (word == "." || word == "/") {
            return self.lookup(if word == "." { "dot" } else { "slash" }, None, Some(-0.5), Some(ctx));
        }
        if let Some(s) = match word {
            "%" => Some("percent"),
            "&" => Some("and"),
            "+" => Some("plus"),
            "@" => Some("at"),
            _ => None,
        } {
            return self.lookup(s, None, None, Some(ctx));
        }
        if word.trim_matches('.').contains('.') && isalpha(&word.replace('.', "")) && word.split('.').map(cp_len).max().unwrap_or(0) < 3 {
            return self.get_nnp(word);
        }
        if word == "a" || word == "A" {
            return Some(if tag == Some("DT") { "ɐ" } else { "ˈA" }.to_string());
        }
        if word == "am" || word == "Am" || word == "AM" {
            if tag_is("NN") {
                return self.get_nnp(word);
            }
            if ctx.future_vowel.is_none() || word != "am" || stress.is_some_and(|s| s > 0.0) {
                return self.gold_plain("am");
            }
            return Some("ɐm".to_string());
        }
        if word == "an" || word == "An" || word == "AN" {
            if word == "AN" && tag_is("NN") {
                return self.get_nnp(word);
            }
            return Some("ɐn".to_string());
        }
        if word == "I" && tag == Some("PRP") {
            return Some(format!("{SECONDARY}I"));
        }
        if (word == "by" || word == "By" || word == "BY") && parent_tag(tag).as_deref() == Some("ADV") {
            return Some("bˈI".to_string());
        }
        if word == "to" || word == "To" || (word == "TO" && (tag == Some("TO") || tag == Some("IN"))) {
            return match ctx.future_vowel {
                None => self.gold_plain("to"),
                Some(true) => Some("tʊ".to_string()),
                Some(false) => Some("tə".to_string()),
            };
        }
        if word == "in" || word == "In" || (word == "IN" && tag != Some("NNP")) {
            let s = if ctx.future_vowel.is_none() || tag != Some("IN") { PRIMARY.to_string() } else { String::new() };
            return Some(format!("{s}ɪn"));
        }
        if word == "the" || word == "The" || (word == "THE" && tag == Some("DT")) {
            return Some(if ctx.future_vowel == Some(true) { "ði" } else { "ðə" }.to_string());
        }
        if tag == Some("IN") && is_vs(word) {
            return self.lookup("versus", None, None, Some(ctx));
        }
        if word == "used" || word == "Used" || word == "USED" {
            let used = self.tables.gold("used")?;
            let pick = |k: &str| match used {
                Stored::Het(v) => v.iter().find(|(key, _)| key == k).and_then(|x| x.1.clone()),
                Stored::Plain(p) => Some(p.clone()),
            };
            return if (tag == Some("VBD") || tag == Some("JJ")) && ctx.future_to { pick("VBD") } else { pick("DEFAULT") };
        }
        None
    }

    fn is_known(&self, word: &str) -> bool {
        if self.het_only.get() {
            return self.het_lower(word);
        }
        self.in_golds(word) || self.in_silvers(word) || matches!(word, "%" | "&" | "+" | "@") || eligible(word)
    }
    fn known_word(&self, wl: &str) -> bool {
        eligible(wl) || self.in_golds(wl) || self.in_silvers(wl)
    }

    // ---------------------------------------------------------------------------------------------------- derivation
    pub fn suffix_s(stem: &str) -> Option<String> {
        let last = last_char(stem)?;
        Some(if "ptkfθ".contains(last) {
            format!("{stem}s")
        } else if "szʃʒʧʤ".contains(last) {
            format!("{stem}ᵻz")
        } else {
            format!("{stem}z")
        })
    }
    pub fn suffix_ed(stem: &str) -> Option<String> {
        let c: Vec<char> = stem.chars().collect();
        let last = *c.last()?;
        Some(if "pkfθʃsʧ".contains(last) {
            format!("{stem}t")
        } else if last == 'd' {
            format!("{stem}ᵻd")
        } else if last != 't' {
            format!("{stem}d")
        } else if c.len() < 2 {
            format!("{stem}ɪd")
        } else if US_TAUS.contains(c[c.len() - 2]) {
            format!("{}ɾᵻd", c[..c.len() - 1].iter().collect::<String>())
        } else {
            format!("{stem}ᵻd")
        })
    }
    pub fn suffix_ing(stem: &str) -> Option<String> {
        let c: Vec<char> = stem.chars().collect();
        c.last()?;
        if c.len() > 1 && c[c.len() - 1] == 't' && US_TAUS.contains(c[c.len() - 2]) {
            return Some(format!("{}ɾɪŋ", c[..c.len() - 1].iter().collect::<String>()));
        }
        Some(format!("{stem}ɪŋ"))
    }

    fn stem_s(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<&Ctx>) -> Option<String> {
        if cp_len(word) < 3 || !word.ends_with('s') {
            return None;
        }
        let stem = if !word.ends_with("ss") && self.is_known(&word[..word.len() - 1]) {
            word[..word.len() - 1].to_string()
        } else if (word.ends_with("'s") || (cp_len(word) > 4 && word.ends_with("es") && !word.ends_with("ies"))) && self.is_known(&word[..word.len() - 2]) {
            word[..word.len() - 2].to_string()
        } else if cp_len(word) > 4 && word.ends_with("ies") && self.is_known(&format!("{}y", &word[..word.len() - 3])) {
            format!("{}y", &word[..word.len() - 3])
        } else {
            return None;
        };
        Self::suffix_s(&self.lookup(&stem, tag, stress, ctx)?)
    }
    fn stem_ed(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<&Ctx>) -> Option<String> {
        if cp_len(word) < 4 || !word.ends_with('d') {
            return None;
        }
        let stem = if !word.ends_with("dd") && self.is_known(&word[..word.len() - 1]) {
            word[..word.len() - 1].to_string()
        } else if cp_len(word) > 4 && word.ends_with("ed") && !word.ends_with("eed") && self.is_known(&word[..word.len() - 2]) {
            word[..word.len() - 2].to_string()
        } else {
            return None;
        };
        Self::suffix_ed(&self.lookup(&stem, tag, stress, ctx)?)
    }
    fn stem_ing(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<&Ctx>) -> Option<String> {
        if cp_len(word) < 5 || !word.ends_with("ing") {
            return None;
        }
        let cut = &word[..word.len() - 3];
        let stem = if cp_len(word) > 5 && self.is_known(cut) {
            cut.to_string()
        } else if self.is_known(&format!("{cut}e")) {
            format!("{cut}e")
        } else if cp_len(word) > 5 && stem_pattern(word) && self.is_known(&word[..word.len() - 4]) {
            word[..word.len() - 4].to_string()
        } else {
            return None;
        };
        Self::suffix_ing(&self.lookup(&stem, tag, stress, ctx)?)
    }
    /// misaki's own -s/-ed/-ing rules, only where the stem is one of the stored heteronyms.
    fn derive_het(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<&Ctx>) -> Option<String> {
        self.het_only.set(true);
        let r = self
            .stem_s(word, tag, stress, ctx)
            .or_else(|| self.stem_ed(word, tag, stress, ctx))
            .or_else(|| self.stem_ing(word, tag, Some(stress.unwrap_or(0.5)), ctx));
        self.het_only.set(false);
        r
    }

    // ------------------------------------------------------------------------------------------------ the word cascade
    fn get_word(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: &Ctx) -> Option<String> {
        if let Some(ps) = self.special_case(word, tag, stress, ctx) {
            return Some(ps);
        }
        let wl = word.to_lowercase();
        let mut word = word.to_string();
        let rest = &word[first_char(&word).len_utf8()..];
        if cp_len(&word) > 1 && isalpha(&word.replace('\'', "")) && word != wl && (tag != Some("NNP") || cp_len(&word) > 7) && !self.in_golds(&word) && !self.in_silvers(&word)
            && (word == word.to_uppercase() || rest == rest.to_lowercase()) && self.known_word(&wl)
        {
            word = wl;
        }
        if self.is_known(&word) {
            return self.lookup(&word, tag, stress, Some(ctx));
        }
        let c = Some(ctx);
        if word.ends_with("s'") && self.is_known(&format!("{}'s", &word[..word.len() - 2])) {
            return self.lookup(&format!("{}'s", &word[..word.len() - 2]), tag, stress, c);
        }
        if word.ends_with('\'') && self.is_known(&word[..word.len() - 1]) {
            return self.lookup(&word[..word.len() - 1], tag, stress, c);
        }
        self.stem_s(&word, tag, stress, c).or_else(|| self.stem_ed(&word, tag, stress, c)).or_else(|| self.stem_ing(&word, tag, Some(stress.unwrap_or(0.5)), c))
    }

    /// Stored golds, stored silvers, else (derivation from a heteronym stem, else) the network.
    pub fn lookup(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<&Ctx>) -> Option<String> {
        let mut word = word.to_string();
        let mut is_nnp = false;
        if word == word.to_uppercase() && !self.in_golds(&word) {
            word = word.to_lowercase();
            is_nnp = tag == Some("NNP");
        }
        let mut ps: Option<Stored> = self.stored(false, &word).cloned();
        if ps.is_none() && !is_nnp {
            ps = self.stored(true, &word).cloned();
        }
        if ps.is_none() && eligible(&word) && (!is_nnp || !self.spell_unknown_caps()) {
            let lower = word.to_lowercase();
            if let Some(r) = self.derive_het(&lower, tag, stress, ctx) {
                return Some(r);
            }
            ps = self.net_ps(&lower).map(Stored::Plain);
        }
        let resolved: Option<String> = match ps {
            Some(Stored::Het(_)) => {
                let h = ps.as_ref().unwrap();
                let tag: Option<String> = if ctx.is_some_and(|c| c.future_vowel.is_none()) && h.het_get("None").is_some() {
                    Some("None".to_string())
                } else if tag.is_none() || h.het_get(tag.unwrap()).is_none() {
                    parent_tag(tag)
                } else {
                    tag.map(str::to_string)
                };
                match tag.as_deref().and_then(|t| h.het_get(t)) {
                    Some(v) => v.clone(),
                    None => h.het_get("DEFAULT").and_then(|v| v.clone()),
                }
            }
            Some(Stored::Plain(p)) => Some(p),
            None => None,
        };
        if resolved.as_deref().is_none_or(|p| is_nnp && !p.contains(PRIMARY)) {
            if let Some(r) = self.get_nnp(&word) {
                return Some(r);
            }
        }
        apply_stress(resolved.as_deref(), stress)
    }

    // ------------------------------------------------------------------------------------------------------- numbers
    fn is_currency(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        if word.matches('.').count() > 1 {
            return false;
        }
        word.split('.').nth(1).is_some_and(|f| cp_len(f) < 3)
    }

    fn extend_num(&self, result: &mut Vec<String>, num: Option<String>, first: bool, escape: bool, flags: &str) {
        let words = if escape { num } else { num.and_then(|n| cardinal(&Big::parse(&n))) };
        let Some(words) = words else { return };
        let splits: Vec<&str> = words.split(|c: char| !c.is_ascii_lowercase()).filter(|s| !s.is_empty()).collect();
        for (i, w) in splits.iter().enumerate() {
            if *w != "and" || flags.contains('&') {
                if first && i == 0 && splits.len() > 1 && *w == "one" && flags.contains('a') {
                    result.push("ə".to_string());
                } else {
                    result.push(self.lookup(w, None, if *w == "point" { Some(-2.0) } else { None }, None).unwrap_or_default());
                }
            } else if *w == "and" && flags.contains('n') && !result.is_empty() {
                result.last_mut().unwrap().push_str("ən");
            }
        }
    }

    fn get_number(&self, word: &str, currency: Option<char>, is_head: bool, flags: &str) -> Option<String> {
        let mut word = word.to_string();
        let tail = word.chars().rev().take_while(|c| c.is_ascii_lowercase() || *c == '\'').count();
        let suffix: Option<String> = if tail > 0 { Some(word[word.len() - tail..].to_string()) } else { None };
        if tail > 0 {
            word.truncate(word.len() - tail);
        }
        let mut result: Vec<String> = Vec::new();
        if word.starts_with('-') {
            result.push(self.lookup("minus", None, None, None).unwrap_or_default());
            word.remove(0);
        }
        let suffix_is_ordinal = suffix.as_deref().is_some_and(|s| ORDINALS.contains(&s));
        let ext = |s: &Self, r: &mut Vec<String>, n: &str, first: bool| s.extend_num(r, Some(n.to_string()), first, false, flags);
        if is_digit(&word) && suffix_is_ordinal {
            self.extend_num(&mut result, ordinal(&Big::parse(&word)), true, true, flags);
        } else if result.is_empty() && cp_len(&word) == 4 && !currency.is_some_and(|c| currency_units(c).is_some()) && is_digit(&word) {
            self.extend_num(&mut result, year(&Big::parse(&word)), true, true, flags);
        } else if !is_head && !word.contains('.') {
            let num = word.replace(',', "");
            if num.starts_with('0') || cp_len(&num) > 3 {
                for n in num.chars() {
                    ext(self, &mut result, &n.to_string(), false);
                }
            } else if cp_len(&num) == 3 && !num.ends_with("00") {
                ext(self, &mut result, &num[..1], true);
                if &num[1..2] == "0" {
                    result.push(self.lookup("O", None, Some(-2.0), None).unwrap_or_default());
                    ext(self, &mut result, &num[2..], false);
                } else {
                    ext(self, &mut result, &num[1..], false);
                }
            } else {
                ext(self, &mut result, &num, true);
            }
        } else if word.matches('.').count() > 1 || !is_head {
            let mut first = true;
            for num in word.replace(',', "").split('.') {
                if num.is_empty() {
                } else if num.starts_with('0') || (cp_len(num) != 2 && num.chars().skip(1).any(|n| n != '0')) {
                    for n in num.chars() {
                        ext(self, &mut result, &n.to_string(), false);
                    }
                } else {
                    ext(self, &mut result, num, first);
                }
                first = false;
            }
        } else if let Some(units) = currency.and_then(currency_units).filter(|_| Self::is_currency(&word)) {
            let cleaned = word.replace(',', "");
            let parts: Vec<&str> = cleaned.split('.').collect();
            let unit_names = [units.0, units.1];
            let mut pairs: Vec<(Big, &str)> = (0..parts.len().min(2)).map(|k| (if parts[k].is_empty() { Big::parse("0") } else { Big::parse(parts[k]) }, unit_names[k])).collect();
            let zero = Big::parse("0");
            if pairs.len() > 1 {
                if pairs[1].0 == zero {
                    pairs.truncate(1);
                } else if pairs[0].0 == zero {
                    pairs.remove(0);
                }
            }
            for (i, (num, unit)) in pairs.iter().enumerate() {
                if i > 0 {
                    result.push(self.lookup("and", None, None, None).unwrap_or_default());
                }
                self.extend_num(&mut result, cardinal(num), i == 0, true, flags);
                result.push(if *num != Big::parse("1") && *unit != "pence" { self.stem_s(&format!("{unit}s"), None, None, None) } else { self.lookup(unit, None, None, None) }.unwrap_or_default());
            }
        } else {
            let words = if is_digit(&word) {
                cardinal(&Big::parse(&word))
            } else if !word.contains('.') {
                let n = Big::parse(&word.replace(',', ""));
                if suffix_is_ordinal { ordinal(&n) } else { cardinal(&n) }
            } else {
                word = word.replace(',', "");
                if word.starts_with('.') {
                    let digits: Option<Vec<String>> = word[1..].chars().map(|n| cardinal(&Big::parse(&n.to_string()))).collect();
                    digits.map(|d| format!("point {}", d.join(" ")))
                } else {
                    float_words(&word)
                }
            };
            self.extend_num(&mut result, words, true, true, flags);
        }
        if result.is_empty() {
            return None;
        }
        let ps = result.join(" ");
        match suffix.as_deref() {
            Some("s") | Some("'s") => Self::suffix_s(&ps),
            Some("ed") | Some("'d") => Self::suffix_ed(&ps),
            Some("ing") => Self::suffix_ing(&ps),
            _ => Some(ps),
        }
    }

    fn append_currency(&self, ps: String, currency: Option<char>) -> String {
        let Some(c) = currency else { return ps };
        let unit = currency_units(c).and_then(|u| self.stem_s(&format!("{}s", u.0), None, None, None));
        match unit {
            Some(u) if !u.is_empty() => format!("{ps} {u}"),
            _ => ps,
        }
    }

    fn is_number(word: &str, is_head: bool) -> bool {
        if !word.chars().any(|c| c.is_ascii_digit()) {
            return false;
        }
        let mut w = word;
        for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
            if let Some(r) = w.strip_suffix(s) {
                w = r;
                break;
            }
        }
        w.chars().enumerate().all(|(i, c)| c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))
    }

    /// A word whose spelling ends in é (the final-é rule): the stored words are skipped and the network reads the stem with the é as "ay", so that
    /// "résumé" is not the verb "resume". `stored` is what the stored words give for the plain spelling; when that already ends in /eɪ/ (café) it stays.
    fn final_acute(&self, accented: &str, stress: Option<f64>, stored: Option<&str>) -> Option<String> {
        if stored.is_some_and(|p| p.ends_with('A')) {
            return None;
        }
        let stem: String = accented.chars().take(accented.chars().count() - 1).collect();
        let suffix = if stem.chars().any(char::is_lowercase) || stem.is_empty() { "ay" } else { "AY" }; // the shape of the respelling is the word's, as for any word the network is asked
        let respelled = format!("{}{suffix}", fold_word(&stem).unwrap_or(stem));
        if !eligible(&respelled) {
            return None;
        }
        apply_stress(self.net_ps(&respelled.to_lowercase()).as_deref(), stress)
    }

    /// Lexicon.__call__: phonemes for one (possibly merged) token, or None.
    pub fn call(&self, tk: &Token, ctx: &Ctx) -> Option<String> {
        let raw = tk.u.alias.as_deref().unwrap_or(&tk.text).replace(['‘', '’'], "'");
        let accented: String = nfkc(&raw).chars().map(|c| decimal_value(c).map_or(c, |v| char::from_digit(v, 10).unwrap())).collect();
        let word = fold_word(&accented).unwrap_or_else(|| accented.clone()); // accented spellings are read as their plain ASCII spelling 
        let stress = if word == word.to_lowercase() { None } else { Some(if word == word.to_uppercase() { 2.0 } else { 0.5 }) };
        let stored = self.get_word(&word, Some(&tk.tag), stress, ctx);
        let found = if accented.ends_with(['é', 'É']) { self.final_acute(&accented, stress, stored.as_deref()) } else { None }.or(stored);
        if let Some(ps) = found {
            return apply_stress(Some(&self.append_currency(ps, tk.u.currency)), tk.u.stress);
        }
        if Self::is_number(&word, tk.u.is_head) {
            let ps = self.get_number(&word, tk.u.currency, tk.u.is_head, &tk.u.num_flags);
            return apply_stress(ps.as_deref(), tk.u.stress);
        }
        None
    }
}

/// /^vs\.?\n?$/i
fn is_vs(word: &str) -> bool {
    let w = word.to_lowercase();
    let w = w.strip_suffix('\n').unwrap_or(&w);
    let w = w.strip_suffix('.').unwrap_or(w);
    w == "vs"
}
