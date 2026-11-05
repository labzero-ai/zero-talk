//! The stored word tables of the G2P route (the `tables` section of the front-end file): exceptions to the network, heteronyms
//! and the directly read letters. Sections start with a line "#name"; entries are front-coded in sorted order: one digit (the number of
//! leading characters shared with the previous key), the rest of the key, a tab, the value (a phoneme string; heteronyms: a JSON
//! object of tag to phonemes or null).
use std::collections::HashMap;

use crate::archive::Result;

#[derive(Clone, Debug)]
pub enum Stored {
    Plain(String),
    /// Tag to phonemes (null: the tag is listed but has no phonemes); "DEFAULT" is always there.
    Het(Vec<(String, Option<String>)>),
}

impl Stored {
    pub fn is_het(&self) -> bool {
        matches!(self, Stored::Het(_))
    }
    /// The entry listed under `tag`, outer None if the tag is not listed.
    pub fn het_get(&self, tag: &str) -> Option<&Option<String>> {
        match self {
            Stored::Het(v) => v.iter().find(|(k, _)| k == tag).map(|(_, p)| p),
            Stored::Plain(_) => None,
        }
    }
}

pub struct Tables {
    gold: HashMap<String, Stored>,
    silver: HashMap<String, Stored>,
}

/// {"K": "v", "K2": null}: strings without escapes, as the front-end file writes them.
fn parse_het(s: &str) -> Result<Vec<(String, Option<String>)>> {
    let bad = || format!("bad heteronym entry {s:?}");
    let body = s.strip_prefix('{').and_then(|r| r.strip_suffix('}')).ok_or_else(bad)?;
    let mut out = Vec::new();
    let mut rest = body.trim();
    while !rest.is_empty() {
        let r = rest.strip_prefix('"').ok_or_else(bad)?;
        let (key, r) = r.split_once('"').ok_or_else(bad)?;
        let r = r.trim_start().strip_prefix(':').ok_or_else(bad)?.trim_start();
        let (val, r) = if let Some(r) = r.strip_prefix("null") {
            (None, r)
        } else {
            let r = r.strip_prefix('"').ok_or_else(bad)?;
            let (v, r) = r.split_once('"').ok_or_else(bad)?;
            (Some(v.to_string()), r)
        };
        out.push((key.to_string(), val));
        rest = r.trim_start().strip_prefix(',').unwrap_or(r).trim_start();
    }
    Ok(out)
}

impl Tables {
    pub fn parse(text: &str) -> Result<Self> {
        let (mut gold, mut silver) = (HashMap::new(), HashMap::new());
        let mut section = "";
        let mut prev = String::new();
        for line in text.split('\n').filter(|l| !l.is_empty()) {
            if let Some(name) = line.strip_prefix('#') {
                section = name;
                prev.clear();
                continue;
            }
            let (head, val) = line.split_once('\t').ok_or_else(|| format!("bad table line {line:?}"))?;
            let n = head.chars().next().and_then(|c| c.to_digit(10)).ok_or_else(|| format!("bad table line {line:?}"))? as usize;
            let key: String = prev.chars().take(n).chain(head[1..].chars()).collect();
            let entry = if val.starts_with('{') { Stored::Het(parse_het(val)?) } else { Stored::Plain(val.to_string()) };
            match section {
                "exc-gold" | "het" | "direct" => gold.insert(key.clone(), entry),
                "exc-silver" => silver.insert(key.clone(), entry),
                other => return Err(format!("unknown table section {other:?}")),
            };
            prev = key;
        }
        Ok(Tables { gold, silver })
    }
    pub fn gold(&self, key: &str) -> Option<&Stored> {
        self.gold.get(key)
    }
    pub fn silver(&self, key: &str) -> Option<&Stored> {
        self.silver.get(key)
    }
}
