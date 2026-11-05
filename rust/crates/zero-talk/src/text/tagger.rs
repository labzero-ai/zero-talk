//! The tagger: rules for white space, punctuation, numbers and acronyms, and a targeted averaged perceptron that tags only the
//! words whose tag changes misaki's output (the heteronyms, "that", "in", "used"), from a window of lower-cased neighbours. The shipped
//! data is the targeted kind, which is all this reads.
use std::collections::{HashMap, HashSet};

use super::pyutil::*;

pub struct Tagger {
    tags: Vec<String>,
    /// Tag indices in ascending order of the tag's name (ties go to the greatest name).
    order: Vec<usize>,
    ctx: HashSet<String>,
    targets: HashSet<String>,
    weights: HashMap<String, Vec<(usize, i32)>>,
}

fn upper(c: char) -> bool {
    c.to_uppercase().eq(std::iter::once(c)) && !c.to_lowercase().eq(std::iter::once(c))
}
fn tail(s: &str, n: usize) -> String {
    let c: Vec<char> = s.chars().collect();
    c[c.len().saturating_sub(n)..].iter().collect()
}

/// The regular-expression class \s.
fn js_space(c: char) -> bool {
    matches!(c, '\t'..='\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}')
}

const OPENERS: [&str; 9] = ["(", "[", "{", "—", "–", "-", "/", "“", "‘"];

impl Tagger {
    pub fn load(text: &str) -> Result<Tagger, String> {
        let (mut kind, mut tags, mut ctx, mut targets) = (String::new(), Vec::new(), HashSet::new(), HashSet::new());
        let mut weights = HashMap::new();
        for line in text.split('\n').filter(|l| !l.is_empty()) {
            if let Some(v) = line.strip_prefix("#kind ") {
                kind = v.to_string();
            } else if let Some(v) = line.strip_prefix("#targets ") {
                targets = v.split(' ').map(str::to_string).collect();
            } else if let Some(v) = line.strip_prefix("#tags ") {
                tags = v.split(' ').map(str::to_string).collect();
            } else if let Some(v) = line.strip_prefix("#ctx ") {
                ctx = v.split(' ').map(str::to_string).collect();
            } else if line.starts_with('#') {
                // #scale: scores are compared as integers, so the scale is not needed
            } else {
                let tab = line.rfind('\t').ok_or("bad tagger line")?;
                let pairs = line[tab + 1..]
                    .split(',')
                    .map(|p| {
                        let (a, b) = p.split_once(':').ok_or("bad tagger weight")?;
                        Ok((a.parse::<usize>().map_err(|e| e.to_string())?, b.parse::<i32>().map_err(|e| e.to_string())?))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                weights.insert(line[..tab].to_string(), pairs);
            }
        }
        if kind != "targeted" {
            return Err(format!("tagger model of kind {kind:?}: only the targeted kind is supported"));
        }
        let mut order: Vec<usize> = (0..tags.len()).collect();
        order.sort_by(|&a, &b| tags[a].cmp(&tags[b]));
        Ok(Tagger { tags, order, ctx, targets, weights })
    }

    /// Tags that need no model: white space, punctuation (quotes by position), numbers and, in mixed-case text, acronyms.
    fn rule_tag(toks: &[(String, String)], i: usize, sentence_upper: bool) -> Option<&'static str> {
        let (t, ws) = (&toks[i].0, &toks[i].1);
        if !t.is_empty() && t.chars().all(js_space) {
            return Some("_SP");
        }
        let prev_space = i == 0 || !toks[i - 1].1.is_empty();
        match t.as_str() {
            "," => return Some(","),
            ":" | ";" | "):" => return Some(":"),
            "(" | "{" => return Some("-LRB-"),
            "[" => return Some("XX"),
            ")" | "]" | "}" => return Some("-RRB-"),
            "“" | "‘" | "«" => return Some("``"),
            "”" | "’" | "»" => return Some("''"),
            "\"" | "'" => return Some(if (prev_space || OPENERS.contains(&toks[i - 1].0.as_str())) && ws.is_empty() { "``" } else { "''" }),
            "$" | "£" | "€" | "#" => return Some("$"),
            "—" | "–" | "--" | "---" | "..." => return Some(":"),
            "…" => return Some("NFP"),
            "%" => return Some("NN"),
            "/" => return Some("SYM"),
            "&" => return Some("CC"),
            "-" => return Some(if i > 0 && toks[i - 1].1.is_empty() && ws.is_empty() { "HYPH" } else { ":" }),
            _ => {}
        }
        if !t.is_empty() && t.chars().all(|c| matches!(c, '.' | '!' | '?')) {
            return Some(".");
        }
        let b = t.as_bytes();
        if b.first().is_some_and(u8::is_ascii_digit) && b.iter().all(|c| c.is_ascii_digit() || *c == b'.' || *c == b',') {
            return Some("CD");
        }
        // /^[0-9]+(?:[:/][0-9]+)+$/
        if t.split([':', '/']).count() > 1 && t.chars().all(|c| c.is_ascii_digit() || c == ':' || c == '/') && t.split([':', '/']).all(|p| !p.is_empty()) {
            return Some("CD");
        }
        if !sentence_upper && t.chars().count() > 1 && t.chars().all(|c| c.is_uppercase() && is_letter(c)) {
            return Some("NNP");
        }
        None
    }

    fn features(&self, toks: &[(String, String)], i: usize) -> Vec<String> {
        let text = &toks[i].0;
        let lw = text.to_lowercase();
        let first = text.chars().next();
        let mut f = vec!["b".to_string(), format!("w={lw}"), format!("sf={}", tail(&lw, 2))];
        f.push(format!("sh={}{}", if first.is_some_and(upper) { "C" } else { "l" }, if i == 0 { "F" } else { "" }));
        let mut ctx = Vec::new();
        for o in [-2i32, -1, 1, 2] {
            let j = i as i32 + o;
            let c = if j < 0 {
                "<s>".to_string()
            } else if j as usize >= toks.len() {
                "</s>".to_string()
            } else {
                let t = toks[j as usize].0.to_lowercase();
                if self.ctx.contains(&t) { t } else { "?".to_string() }
            };
            f.push(format!("c{o}={c}"));
            ctx.push(c);
        }
        for o in [-1i32, 1] {
            let j = i as i32 + o;
            if j >= 0 && (j as usize) < toks.len() {
                f.push(format!("s{o}={}", tail(&toks[j as usize].0.to_lowercase(), 2)));
            }
        }
        f.push(format!("w|c-1={lw}|{}", ctx[1]));
        f.push(format!("w|c+1={lw}|{}", ctx[2]));
        f.push(format!("c-1|c+1={}|{}", ctx[1], ctx[2]));
        f.push(format!("c-2|c-1={}|{}", ctx[0], ctx[1]));
        f.push(format!("c+1|c+2={}|{}", ctx[2], ctx[3]));
        f
    }

    fn sentence_start(toks: &[(String, String)], i: usize) -> bool {
        for j in (0..i).rev() {
            let t = toks[j].0.as_str();
            if matches!(t, "“" | "\"" | "‘" | "'" | "(" | "[" | "{") {
                continue;
            }
            return (!t.is_empty() && t.chars().all(|c| matches!(c, '.' | '!' | '?'))) || t == "”";
        }
        true
    }

    /// toks: (text, trailing whitespace) -> a tag per token.
    pub fn tag(&self, toks: &[(String, String)], sentence_upper: bool) -> Vec<String> {
        let view: Vec<(String, String)> = if sentence_upper { toks.iter().map(|(t, w)| (t.to_lowercase(), w.clone())).collect() } else { toks.to_vec() };
        let mut out = Vec::with_capacity(toks.len());
        for i in 0..toks.len() {
            if let Some(rule) = Self::rule_tag(toks, i, sentence_upper) {
                out.push(rule.to_string());
                continue;
            }
            let t = &view[i].0;
            let lw = t.to_lowercase();
            if self.targets.contains(&lw) {
                let mut s = vec![0i64; self.tags.len()];
                for name in self.features(&view, i) {
                    if let Some(ws) = self.weights.get(&name) {
                        for &(k, v) in ws {
                            s[k] += v as i64;
                        }
                    }
                }
                let mut best = self.order[0];
                for &k in &self.order {
                    if s[k] >= s[best] {
                        best = k;
                    }
                }
                out.push(self.tags[best].clone());
            } else if let Some(d) = default_tag(&lw) {
                out.push(d.to_string());
            } else if !sentence_upper && t.chars().next().is_some_and(upper) && !self.ctx.contains(&lw) && !Self::sentence_start(&view, i) && t.chars().next().is_some_and(is_letter) {
                out.push("NNP".to_string());
            } else {
                out.push("NN".to_string());
            }
        }
        out
    }
}

fn default_tag(lw: &str) -> Option<&'static str> {
    Some(match lw {
        "the" | "a" | "an" => "DT",
        "i" => "PRP",
        "to" => "TO",
        "am" => "VBP",
        "vs" | "vs." | "of" => "IN",
        "n't" => "RB",
        "and" | "or" => "CC",
        _ => return None,
    })
}
