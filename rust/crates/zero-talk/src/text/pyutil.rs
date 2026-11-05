//! Python string semantics that misaki and Kokoro rely on (str.strip, isalpha, capitalize, ...). Rust's character classes stand
//! in for the Unicode property escapes; they agree on the letters, digits and punctuation of ordinary text.

/// str.isspace(): the characters Python treats as white space.
pub fn is_space(c: char) -> bool {
    matches!(c, '\t'..='\r' | '\x1c'..='\x1f' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}
pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_space)
}
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// \p{L}
pub fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !c.is_numeric()
}
/// str.isalpha() / /^\p{L}+$/u
pub fn isalpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_letter)
}
/// is_digit of misaki: /^[0-9]+\n?$/
pub fn is_digit(s: &str) -> bool {
    let s = s.strip_suffix('\n').unwrap_or(s);
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

pub fn first_char(s: &str) -> char {
    s.chars().next().unwrap_or('\0')
}
pub fn last_char(s: &str) -> Option<char> {
    s.chars().next_back()
}
pub fn cp_len(s: &str) -> usize {
    s.chars().count()
}
/// First character upper-cased, the rest lower-cased (Python's str.capitalize()).
pub fn capitalize(s: &str) -> String {
    let mut it = s.chars();
    match it.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().chain(it.as_str().to_lowercase().chars()).collect(),
    }
}
/// s.rsplit(sep, 1)
pub fn rsplit1<'a>(s: &'a str, sep: &str) -> Vec<&'a str> {
    match s.rfind(sep) {
        Some(i) => vec![&s[..i], &s[i + sep.len()..]],
        None => vec![s],
    }
}

/// The value of a decimal digit outside ASCII, for the blocks of digits a text is likely to hold (unicodedata.numeric after NFKC).
pub fn decimal_value(c: char) -> Option<u32> {
    const STARTS: [u32; 6] = [0x0660, 0x06f0, 0x0966, 0x09e6, 0x0e50, 0x0ed0];
    let cp = c as u32;
    STARTS.iter().find(|&&s| (s..s + 10).contains(&cp)).map(|s| cp - s)
}

/// A subset of Unicode NFKC: the compatibility characters an English text can hold (full-width forms, ligatures, the spaces,
/// super- and subscript digits, vulgar fractions, circled digits and letters, the ellipsis, the trade mark sign). It does not compose letters with combining marks.
pub fn nfkc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{ff01}'..='\u{ff5e}' => out.push(char::from_u32(c as u32 - 0xfee0).unwrap()),
            '\u{3000}' | '\u{a0}' | '\u{2000}'..='\u{200a}' | '\u{202f}' | '\u{205f}' => out.push(' '),
            '\u{fb00}' => out.push_str("ff"),
            '\u{fb01}' => out.push_str("fi"),
            '\u{fb02}' => out.push_str("fl"),
            '\u{fb03}' => out.push_str("ffi"),
            '\u{fb04}' => out.push_str("ffl"),
            '\u{fb05}' | '\u{fb06}' => out.push_str("st"),
            '\u{2026}' => out.push_str("..."),
            '\u{2122}' => out.push_str("TM"),
            '\u{2070}' => out.push('0'),
            '\u{bc}' => out.push_str("1\u{2044}4"),
            '\u{bd}' => out.push_str("1\u{2044}2"),
            '\u{be}' => out.push_str("3\u{2044}4"),
            '\u{207b}' => out.push('\u{2212}'),
            '\u{2460}'..='\u{2473}' => out.push_str(&(c as u32 - 0x245f).to_string()),
            '\u{24ea}' => out.push('0'),
            '\u{24b6}'..='\u{24cf}' => out.push(char::from_u32(c as u32 - 0x24b6 + 'A' as u32).unwrap()),
            '\u{24d0}'..='\u{24e9}' => out.push(char::from_u32(c as u32 - 0x24d0 + 'a' as u32).unwrap()),
            '\u{b9}' => out.push('1'),
            '\u{b2}' | '\u{b3}' => out.push(char::from_digit(c as u32 - 0xb0, 10).unwrap()),
            '\u{2074}'..='\u{2079}' => out.push(char::from_digit(c as u32 - 0x2070, 10).unwrap()),
            '\u{2080}'..='\u{2089}' => out.push(char::from_digit(c as u32 - 0x2080, 10).unwrap()),
            _ => out.push(c),
        }
    }
    out
}
