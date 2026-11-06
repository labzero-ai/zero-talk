//! A port of spaCy 3.8's English tokenizer (Tokenizer.__call__, _split_affixes, _attach_tokens). The 1,343 special cases come
//! from en_core_web_sm. The prefix, suffix, infix and URL patterns are spaCy's, written as matchers instead of regular expressions
//! (they use look-behind). Rust's character classes stand in for the Unicode property escapes of letters and digits; the symbol and
//! pictograph class is a generated table.
//! Ported from spaCy 3.8 (MIT), with special cases from en_core_web_sm 3.8.0 (MIT), and changed for this engine.
use std::collections::HashMap;

use super::pyutil::{is_letter, is_space};
use super::symbols::is_icon;

const QUOTES: &str = "'\"”“`‘´’‚,„»«「」『』（）〔〕【】《》〈〉⟦⟧";
const LIST_PUNCT: &[&str] = &["…", "……", ",", ":", ";", "!", "?", "¿", "؟", "¡", "(", ")", "[", "]", "{", "}", "<", ">", "_", "#", "*", "&", "。", "？", "！", "，", "、", "；", "：", "～", "·", "।", "،", "۔", "؛", "٪"];
const LIST_QUOTES_ALT: &[&str] = &["…", "'", "\"", "”", "“", "`", "‘", "´", "’", "‚", ",", "„", "»", "«", "「", "」", "『", "』", "（", "）", "〔", "〕", "【", "】", "《", "》", "〈", "〉", "〈", "〉", "⟦", "⟧"];
const CUR_SYMBOLS: &[&str] = &["$", "£", "€", "¥", "฿", "US$", "C$", "A$", "₽", "﷼", "₴", "₠", "₡", "₢", "₣", "₤", "₥", "₦", "₧", "₨", "₩", "₪", "₫", "€", "₭", "₮", "₯", "₰", "₱", "₲", "₳", "₴", "₵", "₶", "₷", "₸", "₹", "₺", "₻", "₼", "₽", "₾", "₿"];
const UNITS: &[&str] = &["km", "km²", "km³", "m", "m²", "m³", "dm", "dm²", "dm³", "cm", "cm²", "cm³", "mm", "mm²", "mm³", "ha", "µm", "nm", "yd", "in", "ft", "kg", "g", "mg", "µg", "t", "lb", "oz", "m/s", "km/h", "kmh", "mph", "hPa", "Pa", "mbar", "mb", "MB", "kb", "KB", "gb", "GB", "tb", "TB", "T", "G", "M", "K", "%"];
/// The `PUNCT` characters spaCy pastes into the look-behind class of the period suffix (the '|' included).
const SUFFIX_DOT_CLASS: &str = "…|,:;!?¿؟¡()[]{}<>_#*&。？！，、；：～·।،۔؛٪%²-+'\"”“`‘´’‚„»«「」『』（）〔〕【】《》〈〉⟦⟧";

fn is_ll(c: char) -> bool {
    c.is_lowercase() && is_letter(c)
}
fn is_lu(c: char) -> bool {
    c.is_uppercase() && is_letter(c)
}

fn strs(c: &[char]) -> String {
    c.iter().collect()
}

pub struct Tokenizer {
    specials: HashMap<String, Vec<String>>,
    /// For the second pass: the first piece of a special case's plain split to the splits (in file order).
    matcher: HashMap<String, Vec<Vec<String>>>,
}

/// Length in characters of the prefix at the start of `s` (0: none): the first alternative of spaCy's prefix pattern that matches.
fn prefix_len(s: &[char]) -> usize {
    let at = |k: usize| s.get(k).copied();
    let starts = |lit: &str| {
        let l: Vec<char> = lit.chars().collect();
        s.len() >= l.len() && s[..l.len()] == l[..]
    };
    let Some(c0) = at(0) else { return 0 };
    if matches!(c0, '§' | '%' | '=' | '—' | '–') {
        return 1;
    }
    if c0 == '+' && !at(1).is_some_and(|c| c.is_ascii_digit()) {
        return 1;
    }
    for lit in LIST_PUNCT {
        if starts(lit) {
            return lit.chars().count();
        }
    }
    if c0 == '.' && at(1) == Some('.') {
        return s.iter().take_while(|&&c| c == '.').count();
    }
    for lit in LIST_QUOTES_ALT {
        if starts(lit) {
            return lit.chars().count();
        }
    }
    for lit in CUR_SYMBOLS {
        if starts(lit) {
            return lit.chars().count();
        }
    }
    usize::from(is_icon(c0))
}

/// Length in characters of the suffix at the end of `s`: the leftmost start from which one alternative of spaCy's suffix pattern
/// matches up to the end.
fn suffix_len(s: &[char]) -> usize {
    let n = s.len();
    let dots = s.iter().rev().take_while(|&&c| c == '.').count();
    let start_min = n.saturating_sub(6).min(n.saturating_sub(dots));
    for idx in start_min..n {
        let rest = strs(&s[idx..]);
        let prev = |k: usize| if idx >= k { Some(s[idx - k]) } else { None };
        let prev_digit = prev(1).is_some_and(|c| c.is_ascii_digit());
        let r: Vec<char> = s[idx..].to_vec();
        let single = r.len() == 1;
        let ok = LIST_PUNCT.contains(&rest.as_str())
            || (r.len() >= 2 && r.iter().all(|&c| c == '.'))
            || LIST_QUOTES_ALT.contains(&rest.as_str())
            || (single && is_icon(r[0]))
            || matches!(rest.as_str(), "'s" | "'S" | "’s" | "’S" | "—" | "–")
            || (rest == "+" && prev_digit)
            || (rest == "." && prev(1).is_some_and(|c| matches!(c, 'F' | 'f' | 'C' | 'c' | 'K' | 'k')) && prev(2) == Some('°'))
            || (prev_digit && (CUR_SYMBOLS.contains(&rest.as_str()) || UNITS.contains(&rest.as_str())))
            || (rest == "." && prev(1).is_some_and(|c| c.is_ascii_digit() || is_ll(c) || SUFFIX_DOT_CLASS.contains(c)))
            || (rest == "." && prev(1).is_some_and(is_lu) && prev(2).is_some_and(is_lu));
        if ok {
            return n - idx;
        }
    }
    0
}

/// The first infix match at or after `from`: (start, end), by spaCy's infix alternatives in order.
fn find_infix(s: &[char], from: usize) -> Option<(usize, usize)> {
    let n = s.len();
    let at = |k: usize| s.get(k).copied();
    for p in from..n {
        let c = s[p];
        let before = if p > 0 { Some(s[p - 1]) } else { None };
        if c == '.' && at(p + 1) == Some('.') {
            return Some((p, p + s[p..].iter().take_while(|&&x| x == '.').count()));
        }
        if c == '…' || is_icon(c) {
            return Some((p, p + 1));
        }
        if matches!(c, '+' | '-' | '*' | '^') && before.is_some_and(|b| b.is_ascii_digit()) && at(p + 1).is_some_and(|x| x.is_ascii_digit() || x == '-') {
            return Some((p, p + 1));
        }
        if c == '.' && before.is_some_and(|b| is_ll(b) || QUOTES.contains(b)) && at(p + 1).is_some_and(|x| is_lu(x) || QUOTES.contains(x)) {
            return Some((p, p + 1));
        }
        if c == ',' && before.is_some_and(is_letter) && at(p + 1).is_some_and(is_letter) {
            return Some((p, p + 1));
        }
        if before.is_some_and(|b| is_letter(b) || b.is_ascii_digit()) {
            for alt in ["-", "–", "—", "--", "---", "——", "~"] {
                let l: Vec<char> = alt.chars().collect();
                if s.len() >= p + l.len() && s[p..p + l.len()] == l[..] && at(p + l.len()).is_some_and(is_letter) {
                    return Some((p, p + l.len()));
                }
            }
            if matches!(c, ':' | '<' | '>' | '=' | '/') && at(p + 1).is_some_and(is_letter) {
                return Some((p, p + 1));
            }
        }
    }
    None
}

fn js_space(c: char) -> bool {
    is_space(c) || c == '\u{feff}'
}

/// spaCy's URL_MATCH, anchored at both ends, for a span without white space:
/// (scheme://)? (user@)? (IPv4 | (label.)+ tld) (:port)? ([/?#]...)?
fn url_match(s: &[char]) -> bool {
    let n = s.len();
    let at = |k: usize| s.get(k).copied();
    let a = |c: char| c.is_ascii_alphanumeric() || ('\u{a1}'..='\u{ffff}').contains(&c);
    let m = |c: char| a(c) || c == '_' || c == '-';
    let scheme_class = |c: char| is_letter(c) || c.is_numeric() || matches!(c, '_' | '+' | '-' | '.');
    let mut p1 = vec![0usize];
    if let Some(k) = (0..n.saturating_sub(2)).find(|&k| s[k] == ':' && s[k + 1] == '/' && s[k + 2] == '/') {
        if k >= 2 && s[..k].iter().all(|&c| scheme_class(c)) {
            p1.push(k + 3);
        }
    }
    let mut p2 = p1.clone();
    for &p in &p1 {
        for q in p + 1..n {
            if s[q] == '@' && s[p..q].iter().all(|&c| !js_space(c)) {
                p2.push(q + 1);
            }
        }
    }
    // (:\d{2,5})?([/?#]\S*)? up to the end, from position h
    let tail_ok = |h: usize| -> bool {
        let rest_ok = |r: usize| r == n || matches!(s[r], '/' | '?' | '#');
        let digits = if at(h) == Some(':') { s[h + 1..].iter().take_while(|c| c.is_ascii_digit()).count() } else { 0 };
        rest_ok(h) || (2..=digits.min(5)).any(|d| rest_ok(h + 1 + d))
    };
    let digit_run = |p: usize| if p < n { s[p..].iter().take_while(|c| c.is_ascii_digit()).count() } else { 0 };
    // (\.\d{1,3}){count} as a prefix of s[p..]
    let dotted = |mut p: usize, count: usize| -> bool {
        for g in 0..count {
            if at(p) != Some('.') {
                return false;
            }
            let r = digit_run(p + 1);
            if r == 0 || (g + 1 < count && r > 3) {
                return false;
            }
            p += 1 + r.min(3);
        }
        true
    };
    let lit = |p: usize, l: &str| {
        let l: Vec<char> = l.chars().collect();
        p + l.len() <= n && s[p..p + l.len()] == l[..]
    };
    // The octet patterns, for a run of exactly `len` digits at p.
    let octet = |p: usize, len: usize, kind: u8| -> bool {
        if p + len > n || !s[p..p + len].iter().all(|c| c.is_ascii_digit()) {
            return false;
        }
        let d: Vec<u32> = s[p..p + len].iter().map(|c| c.to_digit(10).unwrap()).collect();
        match (kind, len) {
            (0 | 2, 1) => d[0] >= 1,
            (1, 1) => true,
            (0 | 2, 2) => d[0] >= 1,
            (1, 2) => true,
            (0, 3) => d[0] == 1 || (d[0] == 2 && (d[1] <= 1 || (d[1] == 2 && d[2] <= 3))),
            (1, 3) => d[0] == 1 || (d[0] == 2 && (d[1] <= 4 || (d[1] == 5 && d[2] <= 5))),
            (2, 3) => d[0] == 1 || (d[0] == 2 && (d[1] <= 4 || (d[1] == 5 && d[2] <= 4))),
            _ => false,
        }
    };
    for &start in &p2 {
        let private = ((lit(start, "10") && dotted(start + 2, 3)) || (lit(start, "127") && dotted(start + 3, 3)))
            || ((lit(start, "169.254") || lit(start, "192.168")) && dotted(start + 7, 2))
            || (lit(start, "172.") && digit_run(start + 4) >= 2 && octet(start + 4, 2, 1) && (16..=31).contains(&s[start + 4..start + 6].iter().collect::<String>().parse::<u32>().unwrap_or(0)) && dotted(start + 6, 2));
        if !private {
            for l1 in (1..=3).filter(|&l| octet(start, l, 0)) {
                let p = start + l1;
                if at(p) != Some('.') {
                    continue;
                }
                for l2 in (1..=3).filter(|&l| octet(p + 1, l, 1)) {
                    let p = p + 1 + l2;
                    if at(p) != Some('.') {
                        continue;
                    }
                    for l3 in (1..=3).filter(|&l| octet(p + 1, l, 1)) {
                        let p = p + 1 + l3;
                        if at(p) != Some('.') {
                            continue;
                        }
                        if (1..=3).any(|l4| octet(p + 1, l4, 2) && tail_ok(p + 1 + l4)) {
                            return true;
                        }
                    }
                }
            }
        }
        // (label '.')+ tld, the tld being 2 to 63 lower-case letters
        let mut pos = start;
        while let Some(dot) = (pos..n).find(|&k| s[k] == '.') {
            let label = &s[pos..dot];
            if label.is_empty() || label.len() > 64 || !a(label[0]) || !a(label[label.len() - 1]) || !label.iter().all(|&c| m(c)) {
                break;
            }
            pos = dot + 1;
            let run = s[pos..].iter().take_while(|&&c| is_ll(c)).count();
            if (2..=63).contains(&run) && tail_ok(pos + run) {
                return true;
            }
        }
    }
    false
}

impl Tokenizer {
    pub fn load(exceptions: &str) -> Tokenizer {
        let mut specials = HashMap::new();
        let mut keys = Vec::new();
        for line in exceptions.split('\n').filter(|l| !l.is_empty()) {
            let pieces: Vec<String> = line.split('\t').map(str::to_string).collect();
            let key = pieces.concat();
            if !specials.contains_key(&key) {
                keys.push(key.clone());
            }
            specials.insert(key, pieces);
        }
        let mut t = Tokenizer { specials, matcher: HashMap::new() };
        let mut matcher: HashMap<String, Vec<Vec<String>>> = HashMap::new();
        for key in &keys {
            let kc: Vec<char> = key.chars().collect();
            let infixed = find_infix(&kc, 0).is_some();
            if !(prefix_len(&kc) > 0 || suffix_len(&kc) > 0 || infixed || key.contains(' ')) {
                continue;
            }
            let pat = t.tokenize_span(key, false);
            matcher.entry(pat[0].clone()).or_default().push(pat);
        }
        t.matcher = matcher;
        t
    }

    /// Splits one white-space-free span into tokens (prefixes, the core with infixes, suffixes).
    fn tokenize_span(&self, span: &str, with_specials: bool) -> Vec<String> {
        if with_specials {
            if let Some(p) = self.specials.get(span) {
                return p.clone();
            }
        }
        let mut s: Vec<char> = span.chars().collect();
        let (mut prefixes, mut suffixes): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
        let mut last_size = 0;
        let is_special = |c: &[char]| with_specials && self.specials.contains_key(&strs(c));
        while !s.is_empty() && s.len() != last_size {
            if is_special(&s) {
                break;
            }
            last_size = s.len();
            let pre = prefix_len(&s);
            let (mut prefix, mut minus_pre) = (Vec::new(), Vec::new());
            if pre != 0 {
                prefix = s[..pre].to_vec();
                minus_pre = s[pre..].to_vec();
                if !minus_pre.is_empty() && is_special(&minus_pre) {
                    s = minus_pre;
                    prefixes.push(strs(&prefix));
                    break;
                }
            }
            let suf = suffix_len(&s[pre..]);
            let (mut suffix, mut minus_suf) = (Vec::new(), Vec::new());
            if suf != 0 {
                suffix = s[s.len() - suf..].to_vec();
                minus_suf = s[..s.len() - suf].to_vec();
                if !minus_suf.is_empty() && is_special(&minus_suf) {
                    s = minus_suf;
                    suffixes.push(strs(&suffix));
                    break;
                }
            }
            if pre > 0 && suf > 0 && pre + suf <= s.len() {
                s = s[pre..s.len() - suf].to_vec();
                prefixes.push(strs(&prefix));
                suffixes.push(strs(&suffix));
            } else if pre > 0 {
                s = minus_pre;
                prefixes.push(strs(&prefix));
            } else if suf > 0 {
                s = minus_suf;
                suffixes.push(strs(&suffix));
            }
        }
        let mut out = prefixes;
        if !s.is_empty() {
            if let Some(p) = if with_specials { self.specials.get(&strs(&s)) } else { None } {
                out.extend(p.iter().cloned());
            } else if url_match(&s) {
                out.push(strs(&s));
            } else {
                let mut start = 0;
                let mut from = 0;
                while let Some((a, e)) = find_infix(&s, from) {
                    from = e;
                    if a == start {
                        continue;
                    }
                    out.push(strs(&s[start..a]));
                    if a != e {
                        out.push(strs(&s[a..e]));
                    }
                    start = e;
                }
                if start < s.len() {
                    out.push(strs(&s[start..]));
                }
            }
        }
        out.extend(suffixes.into_iter().rev());
        out
    }

    /// spaCy's second pass: a special case whose key has affixes (or a space) is also found in the middle of a longer chunk.
    fn apply_special_cases(&self, toks: Vec<(String, String)>) -> Vec<(String, String)> {
        struct Span {
            start: usize,
            end: usize,
        }
        let mut spans: Vec<Span> = Vec::new();
        for (i, (t, _)) in toks.iter().enumerate() {
            for pat in self.matcher.get(t).into_iter().flatten() {
                if i + pat.len() > toks.len() || !pat.iter().enumerate().all(|(k, p)| toks[i + k].0 == *p) {
                    continue;
                }
                if !spans.iter().any(|x| x.start == i && x.end == i + pat.len()) {
                    spans.push(Span { start: i, end: i + pat.len() });
                }
            }
        }
        if spans.is_empty() {
            return toks;
        }
        spans.sort_by(|a, b| (b.end - b.start).cmp(&(a.end - a.start)).then(a.start.cmp(&b.start)));
        let mut seen = std::collections::HashSet::new();
        let mut kept: Vec<Span> = Vec::new();
        for m in spans {
            let free = !seen.contains(&m.start) && !seen.contains(&(m.end - 1));
            for k in m.start..m.end {
                seen.insert(k);
            }
            if free {
                kept.push(m);
            }
        }
        kept.sort_by_key(|m| m.start);
        let mut out = Vec::new();
        let (mut i, mut k) = (0, 0);
        while i < toks.len() {
            if k < kept.len() && kept[k].start == i {
                let (start, end) = (kept[k].start, kept[k].end);
                k += 1;
                let text: String = toks[start..end].iter().enumerate().map(|(j, (t, w))| if j < end - start - 1 { format!("{t}{w}") } else { t.clone() }).collect();
                match self.specials.get(&text) {
                    Some(rule) => {
                        for (j, p) in rule.iter().enumerate() {
                            out.push((p.clone(), if j == rule.len() - 1 { toks[end - 1].1.clone() } else { String::new() }));
                        }
                    }
                    None => out.extend(toks[start..end].iter().cloned()),
                }
                i = end;
            } else {
                out.push(toks[i].clone());
                i += 1;
            }
        }
        out
    }

    /// Tokenizer.__call__: (text, whitespace) pairs where whitespace is " " after a token followed by exactly one space.
    pub fn tokenize(&self, text: &str) -> Vec<(String, String)> {
        let chars: Vec<char> = text.chars().collect();
        let mut toks: Vec<(String, String)> = Vec::new();
        if chars.is_empty() {
            return toks;
        }
        let mut in_ws = is_space(chars[0]);
        let mut span_start = 0;
        let flush = |toks: &mut Vec<(String, String)>, a: usize, b: usize| {
            for t in self.tokenize_span(&strs(&chars[a..b]), true) {
                toks.push((t, String::new()));
            }
        };
        for i in 0..chars.len() {
            let ws = is_space(chars[i]);
            if ws != in_ws {
                if span_start < i {
                    flush(&mut toks, span_start, i);
                }
                if chars[i] == ' ' {
                    if let Some(last) = toks.last_mut() {
                        last.1 = " ".to_string();
                    }
                    span_start = i + 1;
                } else {
                    span_start = i;
                }
                in_ws = !in_ws;
            }
        }
        if span_start < chars.len() {
            flush(&mut toks, span_start, chars.len());
        }
        self.apply_special_cases(toks)
    }
}

/// Words that make "St." or "St" after a street name a street: suffixes, directions and unit designators (lower case, without
/// a full stop).
const STREET_FOLLOWERS: &[&str] = &[
    "ave", "avenue", "blvd", "boulevard", "rd", "road", "ln", "lane", "ct", "court", "pl", "place", "way", "pkwy", "parkway", "hwy",
    "highway", "ter", "terrace", "cir", "circle", "sq", "square", "trl", "trail", "plz", "plaza", "n", "s", "e", "w", "ne", "nw", "se",
    "sw", "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "apt", "ste", "suite", "unit",
];

/// The reading of a "St." or "St" token: a street, a saint, or the tokenizer's own reading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StReading {
    Street,
    Saint,
    Other,
}

fn starts_upper(t: &str) -> bool {
    t.chars().next().is_some_and(|c| c.is_uppercase() && is_letter(c))
}

fn is_lower_word(t: &str) -> bool {
    t.chars().next().is_some_and(char::is_lowercase) && t.chars().all(is_letter)
}

/// A number as written: digits and commas, with an optional ordinal ending ("42", "1,200", "5th").
fn is_number(t: &str) -> bool {
    let digits = ["st", "nd", "rd", "th"].iter().find_map(|s| t.strip_suffix(*s)).unwrap_or(t);
    digits.starts_with(|c: char| c.is_ascii_digit()) && digits.chars().all(|c| c.is_ascii_digit() || c == ',')
}

/// The reading of the "St." or "St" token at `i`, from the tokens before and after it. After a street name (a capitalised word, a
/// number, or a lower-case word after a number) it is a street, unless a capitalised word follows that is not a street follower;
/// before a capitalised word it is a saint; otherwise it is `Other`.
pub fn st_reading(toks: &[(String, String)], i: usize) -> StReading {
    let prev = i.checked_sub(1).map(|j| toks[j].0.as_str());
    let before_prev = i.checked_sub(2).map(|j| toks[j].0.as_str());
    let next = toks.get(i + 1).map(|t| t.0.as_str());
    let street_name = prev.is_some_and(|p| starts_upper(p) || is_number(p) || (is_lower_word(p) && before_prev.is_some_and(is_number)));
    let capital_next = next.is_some_and(starts_upper);
    let follower = next.is_some_and(|n| STREET_FOLLOWERS.contains(&n.trim_end_matches('.').to_lowercase().as_str()));
    if street_name && (!capital_next || follower) {
        StReading::Street
    } else if capital_next {
        StReading::Saint
    } else {
        StReading::Other
    }
}

/// The tokens with each "St." or "St" read as "Street" or "Saint" where its neighbours say so, before the tagger and the lexicon see
/// them. A street keeps the full stop only at the end of the text, where it ends the sentence.
pub fn st_readings(toks: Vec<(String, String)>) -> Vec<(String, String)> {
    let n = toks.len();
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let (text, ws) = (toks[i].0.as_str(), toks[i].1.as_str());
        if text != "St." && text != "St" {
            out.push(toks[i].clone());
            continue;
        }
        match st_reading(&toks, i) {
            StReading::Street if text == "St." && i + 1 == n => {
                out.push(("Street".to_string(), String::new()));
                out.push((".".to_string(), ws.to_string()));
            }
            StReading::Street => out.push(("Street".to_string(), ws.to_string())),
            StReading::Saint => out.push(("Saint".to_string(), ws.to_string())),
            StReading::Other => out.push(toks[i].clone()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{st_reading, st_readings, StReading};

    /// Words separated by single spaces as tokens: each token is followed by a space except the last.
    fn toks(s: &str) -> Vec<(String, String)> {
        let words: Vec<&str> = s.split(' ').collect();
        words.iter().enumerate().map(|(i, w)| (w.to_string(), String::from(if i + 1 < words.len() { " " } else { "" }))).collect()
    }

    /// The token texts after the readings of "St." and "St".
    fn read(s: &str) -> Vec<String> {
        st_readings(toks(s)).into_iter().map(|t| t.0).collect()
    }

    #[test]
    fn street_after_a_capitalised_name_and_before_a_lower_case_word() {
        assert_eq!(read("He lives on Main St. near the park"), ["He", "lives", "on", "Main", "Street", "near", "the", "park"]);
        assert_eq!(read("Fifth St. and Elm Ave. meet here"), ["Fifth", "Street", "and", "Elm", "Ave.", "meet", "here"]);
        assert_eq!(read("Oak St and Pine St are near each other"), ["Oak", "Street", "and", "Pine", "Street", "are", "near", "each", "other"]);
        assert_eq!(st_reading(&toks("Oak St. and"), 1), StReading::Street);
    }

    #[test]
    fn the_sentence_with_an_address() {
        assert_eq!(
            read("Dr. Smith lives at 42 Oak St. and works for the U.S. Department of Energy ."),
            ["Dr.", "Smith", "lives", "at", "42", "Oak", "Street", "and", "works", "for", "the", "U.S.", "Department", "of", "Energy", "."]
        );
    }

    #[test]
    fn street_at_the_end_of_the_text_keeps_the_full_stop() {
        assert_eq!(read("Turn left on Pine St."), ["Turn", "left", "on", "Pine", "Street", "."]);
        let t = st_readings(toks("Oak St. and Pine St."));
        assert_eq!(t[1], ("Street".to_string(), " ".to_string()));
        assert_eq!(t[4], ("Street".to_string(), String::new()));
        assert_eq!(t[5], (".".to_string(), String::new()));
    }

    #[test]
    fn street_before_a_comma() {
        assert_eq!(read("Oak St , Boston"), ["Oak", "Street", ",", "Boston"]);
        assert_eq!(read("Ash St , Newtown is near"), ["Ash", "Street", ",", "Newtown", "is", "near"]);
    }

    #[test]
    fn street_after_a_number_or_a_number_and_a_word() {
        assert_eq!(read("Walk down 42 St. and turn left"), ["Walk", "down", "42", "Street", "and", "turn", "left"]);
        assert_eq!(read("Walk along 42 oak St. and wait"), ["Walk", "along", "42", "oak", "Street", "and", "wait"]);
        assert_eq!(read("Ring the bell at 5th St. when you arrive"), ["Ring", "the", "bell", "at", "5th", "Street", "when", "you", "arrive"]);
    }

    #[test]
    fn street_before_a_direction_or_a_unit() {
        assert_eq!(read("Please meet me at Main St. North of the river"), ["Please", "meet", "me", "at", "Main", "Street", "North", "of", "the", "river"]);
        assert_eq!(read("Our office is at Elm St. Suite 200 downtown"), ["Our", "office", "is", "at", "Elm", "Street", "Suite", "200", "downtown"]);
    }

    #[test]
    fn saint_before_a_capitalised_word() {
        assert_eq!(read("We visited St. Patrick's Day parade in March"), ["We", "visited", "Saint", "Patrick's", "Day", "parade", "in", "March"]);
        assert_eq!(read("St. Mary's Day is in May"), ["Saint", "Mary's", "Day", "is", "in", "May"]);
        assert_eq!(read("He liked St Mary's school"), ["He", "liked", "Saint", "Mary's", "school"]);
        assert_eq!(read("Visit St. Mary today"), ["Visit", "Saint", "Mary", "today"]);
        assert_eq!(read("Elm St. Mary is rare"), ["Elm", "Saint", "Mary", "is", "rare"]);
    }

    #[test]
    fn other_readings_are_left_to_the_tokenizer() {
        assert_eq!(read("We walk past St. and then go home"), ["We", "walk", "past", "St.", "and", "then", "go", "home"]);
        assert_eq!(read("Yesterday , St. was quiet"), ["Yesterday", ",", "St.", "was", "quiet"]);
        assert_eq!(read("St. and Oak are both on the list"), ["St.", "and", "Oak", "are", "both", "on", "the", "list"]);
        assert_eq!(read("the church of st. mary"), ["the", "church", "of", "st.", "mary"]);
        assert_eq!(read("Stop at the first stone"), ["Stop", "at", "the", "first", "stone"]);
    }
}
