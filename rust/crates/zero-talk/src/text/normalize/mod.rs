//! Text normalisation ahead of the tokenizer. Written forms that the rules would read wrongly (abbreviations, units, ranges, phone
//! numbers, money scales, fractions, symbols) are rewritten as the words a person would say; everything else passes through
//! unchanged. Each stage takes plain text and returns plain text, so the stages can be tested on their own.
mod abbreviations;
mod numeric;
mod web;

/// The text of one line, rewritten for reading aloud.
pub fn normalize(text: &str) -> String {
    let text = abbreviations::rewrite(text);
    let text = numeric::rewrite(&text);
    web::rewrite(&text)
}
