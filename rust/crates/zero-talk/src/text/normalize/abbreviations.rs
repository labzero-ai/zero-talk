//! Abbreviations written as a word and a full stop (titles, roads, months, weekdays, publishing and business shorthand), and
//! repeated punctuation: "Ave." becomes "Avenue", "Dr." after a street name becomes "Drive", "!!" becomes "!".
//!
//! The full stop goes when the sentence goes on after the expanded word, so that the tokenizer does not end the sentence there.
//! It stays when the text ends after the word, or when a capitalised word follows it for the abbreviations that can end a sentence.

/// The text with its abbreviations written out and its repeated punctuation reduced.
pub(super) fn rewrite(text: &str) -> String {
    words(&punctuation(text))
}

/// What an abbreviation needs from its context before it is written out, and how its full stop is kept.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Before a capitalised name ("Prof. Nguyen"). The full stop always goes.
    Name,
    /// "Dr." is "Drive" after a capitalised word or a number, and before a word that is not capitalised. Otherwise it is left
    /// as it is, because a doctor's name follows.
    Drive,
    /// After a capitalised name ("John Hale Jr.").
    AfterName,
    /// After a capitalised street name or a number ("Elm Ave.").
    Street,
    /// Before a number ("No. 5", "p. 12").
    Number,
    /// "Fl." after an ordinal ("2nd Fl."), or before a number that does not follow a comma (a state name does).
    Floor,
    /// Before a number or a capitalised word ("Apt. 4B", "Fig. 3"). The full stop stays only at the end of the text.
    Place,
    /// Before a number, or after a preposition ("Feb. 2020", "in Dec.").
    Month,
    /// A weekday with a comma, a number, "and", "at" or "to" after it, or "on", "and" or "to" before it.
    Weekday,
    /// "Co." and "Corp." after a capitalised word or an ampersand.
    Company,
    /// "ed." after an ordinal ("3rd ed.").
    Edition,
    /// "est." before a number: "established" before a year, "estimated" before any other number.
    Estimate,
    /// Latin and shorthand inside a sentence ("e.g.", "i.e."), in any context. The full stop stays only at the end of the text.
    Inline,
    /// In any context ("etc.", "Inc."). The full stop stays before a capitalised word and at the end of the text.
    Anywhere,
}

/// Abbreviations (with their full stop where they have one), the words they stand for, and the context they need.
const TABLE: &[(&str, &str, Kind)] = &[
    ("Prof.", "Professor", Kind::Name),
    ("Rev.", "Reverend", Kind::Name),
    ("Gen.", "General", Kind::Name),
    ("Gov.", "Governor", Kind::Name),
    ("Sen.", "Senator", Kind::Name),
    ("Rep.", "Representative", Kind::Name),
    ("Hon.", "Honorable", Kind::Name),
    ("Lt.", "Lieutenant", Kind::Name),
    ("Col.", "Colonel", Kind::Name),
    ("Capt.", "Captain", Kind::Name),
    ("Sgt.", "Sergeant", Kind::Name),
    ("Ms.", "Miz", Kind::Name),
    ("Mt.", "Mount", Kind::Name),
    ("Ft.", "Fort", Kind::Name),
    ("Dr.", "Drive", Kind::Drive),
    ("Jr.", "Junior", Kind::AfterName),
    ("Sr.", "Senior", Kind::AfterName),
    ("Esq.", "Esquire", Kind::AfterName),
    ("Ave.", "Avenue", Kind::Street),
    ("Blvd.", "Boulevard", Kind::Street),
    ("Rd.", "Road", Kind::Street),
    ("Ln.", "Lane", Kind::Street),
    ("Ct.", "Court", Kind::Street),
    ("Pl.", "Place", Kind::Street),
    ("Sq.", "Square", Kind::Street),
    ("Cir.", "Circle", Kind::Street),
    ("Hwy.", "Highway", Kind::Place),
    ("Ste.", "Suite", Kind::Place),
    ("Bldg.", "Building", Kind::Place),
    ("Apt.", "Apartment", Kind::Place),
    ("Fig.", "Figure", Kind::Place),
    ("fig.", "figure", Kind::Place),
    ("Vol.", "Volume", Kind::Place),
    ("vol.", "volume", Kind::Place),
    ("Ch.", "Chapter", Kind::Place),
    ("ch.", "chapter", Kind::Place),
    ("Fl.", "Floor", Kind::Floor),
    ("No.", "number", Kind::Number),
    ("p.", "page", Kind::Number),
    ("pp.", "pages", Kind::Number),
    ("Jan.", "January", Kind::Month),
    ("Feb.", "February", Kind::Month),
    ("Mar.", "March", Kind::Month),
    ("Apr.", "April", Kind::Month),
    ("Jun.", "June", Kind::Month),
    ("Jul.", "July", Kind::Month),
    ("Aug.", "August", Kind::Month),
    ("Sep.", "September", Kind::Month),
    ("Sept.", "September", Kind::Month),
    ("Oct.", "October", Kind::Month),
    ("Nov.", "November", Kind::Month),
    ("Dec.", "December", Kind::Month),
    ("Mon.", "Monday", Kind::Weekday),
    ("Tue.", "Tuesday", Kind::Weekday),
    ("Tues.", "Tuesday", Kind::Weekday),
    ("Wed.", "Wednesday", Kind::Weekday),
    ("Thu.", "Thursday", Kind::Weekday),
    ("Thur.", "Thursday", Kind::Weekday),
    ("Thurs.", "Thursday", Kind::Weekday),
    ("Fri.", "Friday", Kind::Weekday),
    ("Sat.", "Saturday", Kind::Weekday),
    ("Sun.", "Sunday", Kind::Weekday),
    ("e.g.", "for example", Kind::Inline),
    ("i.e.", "that is", Kind::Inline),
    ("cf.", "compare", Kind::Inline),
    ("a.k.a.", "also known as", Kind::Inline),
    ("approx.", "approximately", Kind::Inline),
    ("Approx.", "Approximately", Kind::Inline),
    ("ca.", "circa", Kind::Inline),
    ("w/", "with", Kind::Inline),
    ("w/o", "without", Kind::Inline),
    ("etc.", "et cetera", Kind::Anywhere),
    ("ed.", "edition", Kind::Edition),
    ("est.", "established", Kind::Estimate),
    ("Inc.", "Incorporated", Kind::Anywhere),
    ("Ltd.", "Limited", Kind::Anywhere),
    ("Co.", "Company", Kind::Company),
    ("Corp.", "Corporation", Kind::Company),
    ("Bros.", "Brothers", Kind::Inline),
    ("Univ.", "University", Kind::Inline),
    ("Dept.", "Department", Kind::Inline),
    ("Assn.", "Association", Kind::Inline),
];

const MONTH_BEFORE: &[&str] = &[
    "in", "on", "of", "since", "until", "till", "by", "from", "during", "before", "after", "last", "next", "early", "late", "mid",
    "to", "through", "and", "or",
];
const WEEKDAY_BEFORE: &[&str] = &["on", "and", "or", "to", "from", "until", "till", "by", "last", "next", "every", "this", "through"];
const WEEKDAY_AFTER: &[&str] =
    &["and", "or", "at", "to", "through", "night", "morning", "afternoon", "evening", "noon", "midnight", "tonight"];

/// The text cut into runs of whitespace and runs of other characters, in order. Joining the runs gives the text back.
fn pieces(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut space: Option<bool> = None;
    for (i, c) in text.char_indices() {
        let now = c.is_whitespace();
        if space.is_some_and(|before| before != now) {
            out.push(&text[start..i]);
            start = i;
        }
        space = Some(now);
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

fn is_space(piece: &str) -> bool {
    piece.starts_with(char::is_whitespace)
}

/// Each word with its abbreviation written out; the whitespace between the words stays as it is.
fn words(text: &str) -> String {
    let pieces = pieces(text);
    let words: Vec<&str> = pieces.iter().copied().filter(|piece| !is_space(piece)).collect();
    let mut out = String::with_capacity(text.len() + 16);
    let mut n: usize = 0;
    for piece in &pieces {
        if is_space(piece) {
            out.push_str(piece);
            continue;
        }
        let prev = n.checked_sub(1).map(|k| words[k]);
        let next = words.get(n + 1).copied();
        match expand(piece, prev, next) {
            Some(written) => out.push_str(&written),
            None => out.push_str(piece),
        }
        n += 1;
    }
    out
}

/// A word cut into the punctuation before the abbreviation, the abbreviation itself (letters, full stops and slashes) and the rest.
struct Parts<'a> {
    prefix: &'a str,
    core: &'a str,
    suffix: &'a str,
}

fn split_word(word: &str) -> Parts<'_> {
    let start = word.find(char::is_alphanumeric).unwrap_or(word.len());
    let rest = &word[start..];
    let end = rest.find(|c: char| !(c.is_ascii_alphabetic() || c == '.' || c == '/')).unwrap_or(rest.len());
    Parts { prefix: &word[..start], core: &rest[..end], suffix: &rest[end..] }
}

/// The words on either side of an abbreviation, and whether a comma follows it.
struct Around<'a> {
    prev: Option<&'a str>,
    next: Option<&'a str>,
    comma: bool,
}

/// The written-out form of one word, or None when the word is left as it is.
fn expand(word: &str, prev: Option<&str>, next: Option<&str>) -> Option<String> {
    let parts = split_word(word);
    if parts.suffix.starts_with(char::is_alphanumeric) {
        return None;
    }
    let (full, kind) = lookup(parts.core)?;
    let around = Around { prev, next, comma: parts.suffix.starts_with(',') };
    if !fits(kind, &around) {
        return None;
    }
    let full = if kind == Kind::Estimate && !year(next) { "estimated" } else { full };
    let mut out = String::with_capacity(word.len() + full.len());
    out.push_str(parts.prefix);
    out.push_str(full);
    if stop(parts.core, kind, &around) {
        out.push('.');
    }
    out.push_str(parts.suffix);
    Some(out)
}

fn lookup(core: &str) -> Option<(&'static str, Kind)> {
    TABLE.iter().find(|entry| entry.0 == core).map(|entry| (entry.1, entry.2))
}

/// Whether the abbreviation has the context it needs.
fn fits(kind: Kind, around: &Around) -> bool {
    let (prev, next) = (around.prev, around.next);
    match kind {
        Kind::Name => capital(next),
        Kind::Drive => (capital(prev) || digit(prev)) && !capital(next),
        Kind::AfterName => capital(prev),
        Kind::Street => capital(prev) || digit(prev),
        Kind::Number | Kind::Estimate => digit(next),
        Kind::Floor => ordinal(prev) || (digit(next) && !prev.is_some_and(|w| w.ends_with(','))),
        Kind::Place => capital(next) || digit(next),
        Kind::Month => digit(next) || word_in(prev, MONTH_BEFORE),
        Kind::Weekday => around.comma || digit(next) || word_in(next, WEEKDAY_AFTER) || word_in(prev, WEEKDAY_BEFORE),
        Kind::Company => capital(prev) || prev == Some("&"),
        Kind::Edition => ordinal(prev),
        Kind::Inline | Kind::Anywhere => true,
    }
}

/// Whether the full stop of the abbreviation stays after the expanded word.
fn stop(core: &str, kind: Kind, around: &Around) -> bool {
    if !core.ends_with('.') || around.comma {
        return false;
    }
    match kind {
        Kind::Name => false,
        Kind::Inline | Kind::Place => around.next.is_none(),
        _ => around.next.is_none() || capital(around.next),
    }
}

fn first_alnum(word: Option<&str>) -> Option<char> {
    word?.chars().find(|c| c.is_alphanumeric())
}

fn capital(word: Option<&str>) -> bool {
    first_alnum(word).is_some_and(char::is_uppercase)
}

fn digit(word: Option<&str>) -> bool {
    first_alnum(word).is_some_and(|c| c.is_ascii_digit())
}

/// A number with a suffix of "st", "nd", "rd" or "th", as in "3rd".
fn ordinal(word: Option<&str>) -> bool {
    word.and_then(|w| ["st", "nd", "rd", "th"].iter().find_map(|suffix| w.strip_suffix(*suffix)))
        .is_some_and(|stem| !stem.is_empty() && stem.bytes().all(|b| b.is_ascii_digit()))
}

/// A four-digit year.
fn year(word: Option<&str>) -> bool {
    word.is_some_and(|w| {
        let digits = w.trim_matches(|c: char| !c.is_alphanumeric());
        digits.len() == 4 && digits.bytes().all(|b| b.is_ascii_digit())
    })
}

/// Whether the word, without its surrounding punctuation, is one of the list (ignoring case).
fn word_in(word: Option<&str>, list: &[&str]) -> bool {
    word.is_some_and(|w| {
        let bare = w.trim_matches(|c: char| !c.is_alphanumeric());
        list.iter().any(|item| item.eq_ignore_ascii_case(bare))
    })
}

/// Runs of two or more "!" or "?" become one mark, "?" when the run has one. An ellipsis becomes a comma inside a sentence and a
/// full stop at the end of the text.
fn punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let marks = chars[i..].iter().take_while(|&&c| c == '!' || c == '?').count();
        if marks >= 2 {
            out.push(if chars[i..i + marks].contains(&'?') { '?' } else { '!' });
            i += marks;
        } else if let Some(len) = ellipsis_len(&chars, i) {
            ellipsis(&chars, i, len, &mut out);
            i += len;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// The length of the ellipsis that starts at `i`: the character U+2026, or exactly three full stops.
fn ellipsis_len(chars: &[char], i: usize) -> Option<usize> {
    if chars[i] == '\u{2026}' {
        return Some(1);
    }
    let dots = chars[i..].starts_with(&['.', '.', '.']) && (i == 0 || chars[i - 1] != '.') && chars.get(i + 3) != Some(&'.');
    dots.then_some(3)
}

/// Writes what replaces the ellipsis of `len` characters at `start`. Nothing is written where a sentence already ends before or
/// after it, a comma (with a space after it when none follows) is written inside a sentence, and a full stop at the end of the text.
fn ellipsis(chars: &[char], start: usize, len: usize, out: &mut String) {
    let end = start + len;
    let before = out.trim_end().chars().next_back();
    let offset = chars[end..].iter().position(|c| !c.is_whitespace());
    let after = offset.map(|k| chars[end + k]);
    let both_digits = before.is_some_and(|c| c.is_ascii_digit()) && after.is_some_and(|c| c.is_ascii_digit());
    if both_digits {
        out.extend(&chars[start..end]);
        return;
    }
    let ends_already = before.is_none_or(|c| ".!?,;:".contains(c)) || after.is_some_and(|c| "!?.,;:)]}\"'\u{201d}\u{2019}".contains(c));
    if ends_already {
        return;
    }
    if after.is_none() {
        out.push('.');
        return;
    }
    out.push(',');
    if offset == Some(0) {
        out.push(' ');
    }
}

#[cfg(test)]
mod tests {
    use super::{punctuation, rewrite};

    #[test]
    fn dr_is_drive_after_a_street_name_only() {
        assert_eq!(rewrite("We live on Maple Dr. near the school."), "We live on Maple Drive near the school.");
        assert_eq!(rewrite("Her address is 14 Sunny Dr. in Glenvale."), "Her address is 14 Sunny Drive in Glenvale.");
        assert_eq!(rewrite("Dr. Lee lives on Oak Dr. near the mall."), "Dr. Lee lives on Oak Drive near the mall.");
        assert_eq!(rewrite("Call us at Oak Dr."), "Call us at Oak Drive.");
        assert_eq!(rewrite("Dr. Smith will see you at noon."), "Dr. Smith will see you at noon.");
        assert_eq!(rewrite("Ask Dr. Nguyen about the results."), "Ask Dr. Nguyen about the results.");
        assert_eq!(rewrite("The Dr. Who episode aired on Sunday."), "The Dr. Who episode aired on Sunday.");
        assert_eq!(rewrite("Dr"), "Dr");
    }

    #[test]
    fn road_names_keep_the_sentence_end() {
        assert_eq!(rewrite("She lives on Elm Ave. near the park."), "She lives on Elm Avenue near the park.");
        assert_eq!(rewrite("He lives on Elm Ave."), "He lives on Elm Avenue.");
        assert_eq!(rewrite("We live on Elm Ave. The park is near."), "We live on Elm Avenue. The park is near.");
        assert_eq!(rewrite("Take Ferry Rd. to the coast."), "Take Ferry Road to the coast.");
        assert_eq!(rewrite("The farm is on County Rd. 9."), "The farm is on County Road 9.");
        assert_eq!(rewrite("Their house is on Lake Rd., past the river."), "Their house is on Lake Road, past the river.");
        assert_eq!(rewrite("Sunset Blvd. downtown."), "Sunset Boulevard downtown.");
        assert_eq!(rewrite("Turn left on the ave. today."), "Turn left on the ave. today.");
        assert_eq!(rewrite("Ave. is a word."), "Ave. is a word.");
    }

    #[test]
    fn places_titles_and_numbers() {
        assert_eq!(rewrite("We hiked Mt. Hollis last summer."), "We hiked Mount Hollis last summer.");
        assert_eq!(rewrite("We visited Ft. Halvard in May."), "We visited Fort Halvard in May.");
        assert_eq!(rewrite("He lives in Apt. 4B on the second floor."), "He lives in Apartment 4B on the second floor.");
        assert_eq!(rewrite("Please see No. 5 on the list."), "Please see number 5 on the list.");
        assert_eq!(rewrite("Hwy. 9 is closed tonight."), "Highway 9 is closed tonight.");
        assert_eq!(rewrite("The office is on the 2nd Fl. now."), "The office is on the 2nd Floor now.");
        assert_eq!(rewrite("Prof. Nguyen teaches physics."), "Professor Nguyen teaches physics.");
        assert_eq!(rewrite("Ms. Lee reviewed the report."), "Miz Lee reviewed the report.");
        assert_eq!(rewrite("Lt. Cole arrived on Tuesday."), "Lieutenant Cole arrived on Tuesday.");
        assert_eq!(rewrite("John Smith Jr. spoke first."), "John Smith Junior spoke first.");
        assert_eq!(rewrite("Robert Hale Sr. retired in May."), "Robert Hale Senior retired in May.");
        assert_eq!(rewrite("John Hale, Esq., attended the meeting."), "John Hale, Esquire, attended the meeting.");
    }

    #[test]
    fn titles_and_numbers_left_alone() {
        assert_eq!(rewrite("No, thanks."), "No, thanks.");
        assert_eq!(rewrite("No. I won't go."), "No. I won't go.");
        assert_eq!(rewrite("Mr. Brown arrived at noon."), "Mr. Brown arrived at noon.");
        assert_eq!(rewrite("Mrs. Lee called this morning."), "Mrs. Lee called this morning.");
        assert_eq!(rewrite("Prof. nguyen teaches physics."), "Prof. nguyen teaches physics.");
        assert_eq!(rewrite("Glenvale, Fl. 33101"), "Glenvale, Fl. 33101");
        assert_eq!(rewrite("Apt.4B"), "Apt.4B");
    }

    #[test]
    fn weekdays_and_months_in_dates_and_lists() {
        assert_eq!(rewrite("We meet on Mon. and Wed."), "We meet on Monday and Wednesday.");
        assert_eq!(rewrite("The concert is on Fri. night."), "The concert is on Friday night.");
        assert_eq!(rewrite("Open Tues. to Sun. only."), "Open Tuesday to Sunday only.");
        assert_eq!(rewrite("The Tigers play Sat. at home."), "The Tigers play Saturday at home.");
        assert_eq!(rewrite("The lease ends Jan. 31."), "The lease ends January 31.");
        assert_eq!(rewrite("Nov. 5th is the vote."), "November 5th is the vote.");
        assert_eq!(rewrite("We moved in Feb. 2020."), "We moved in February 2020.");
    }

    #[test]
    fn weekday_and_month_words_that_are_not_abbreviations() {
        assert_eq!(rewrite("Look at the Sun."), "Look at the Sun.");
        assert_eq!(rewrite("I sat down by the pool."), "I sat down by the pool.");
        assert_eq!(rewrite("We met in May."), "We met in May.");
        assert_eq!(rewrite("Mar is a common name."), "Mar is a common name.");
        assert_eq!(rewrite("Dec is short for December."), "Dec is short for December.");
    }

    #[test]
    fn latin_and_shorthand_inside_sentences() {
        assert_eq!(rewrite("Bring fruit, e.g. apples or pears."), "Bring fruit, for example apples or pears.");
        assert_eq!(rewrite("Pick any colour, e.g., red, blue, or green."), "Pick any colour, for example, red, blue, or green.");
        assert_eq!(rewrite("The rule is simple, i.e., nothing is free."), "The rule is simple, that is, nothing is free.");
        assert_eq!(rewrite("Coffee w/ milk, please."), "Coffee with milk, please.");
        assert_eq!(rewrite("Tea w/o sugar, please."), "Tea without sugar, please.");
        assert_eq!(rewrite("Compare it, cf. section two."), "Compare it, compare section two.");
        assert_eq!(rewrite("Dan, a.k.a. Danny, spoke first."), "Dan, also known as Danny, spoke first.");
        assert_eq!(rewrite("The barn was built ca. 1900."), "The barn was built circa 1900.");
        assert_eq!(rewrite("We need pens, etc., for the class."), "We need pens, et cetera, for the class.");
        assert_eq!(rewrite("Bring pens, paper, etc. The end."), "Bring pens, paper, et cetera. The end.");
        assert_eq!(rewrite("See Fig. 3 for the result."), "See Figure 3 for the result.");
        assert_eq!(rewrite("The new Vol. 2 is out."), "The new Volume 2 is out.");
        assert_eq!(rewrite("Ch. 4 starts on page 9."), "Chapter 4 starts on page 9.");
        assert_eq!(rewrite("See p. 12 for details."), "See page 12 for details.");
        assert_eq!(rewrite("Read pp. 12-14 tonight."), "Read pages 12-14 tonight.");
        assert_eq!(rewrite("The 3rd ed. is out now."), "The 3rd edition is out now.");
        assert_eq!(rewrite("The house was built in est. 1890."), "The house was built in established 1890.");
        assert_eq!(rewrite("About est. 40 people came."), "About estimated 40 people came.");
    }

    #[test]
    fn latin_and_shorthand_left_alone() {
        assert_eq!(rewrite("It was 12 ft. long."), "It was 12 ft. long.");
        assert_eq!(rewrite("Meet at 5 p.m. today."), "Meet at 5 p.m. today.");
        assert_eq!(rewrite("The temple dates from 44 B.C."), "The temple dates from 44 B.C.");
        assert_eq!(rewrite("Use eg here, not e.g here."), "Use eg here, not e.g here.");
    }

    #[test]
    fn business_shorthand() {
        assert_eq!(rewrite("Acme Inc. makes tools for farms."), "Acme Incorporated makes tools for farms.");
        assert_eq!(rewrite("Brown Ltd. won the contract."), "Brown Limited won the contract.");
        assert_eq!(rewrite("Atlas Corp. reported record profits."), "Atlas Corporation reported record profits.");
        assert_eq!(rewrite("Smith & Co. sells bread."), "Smith & Company sells bread.");
        assert_eq!(rewrite("Gold Bros. sells watches."), "Gold Brothers sells watches.");
        assert_eq!(rewrite("She studies at Univ. College Northvale."), "She studies at University College Northvale.");
        assert_eq!(rewrite("The Dept. of Energy issued the notice."), "The Department of Energy issued the notice.");
    }

    #[test]
    fn business_words_left_alone() {
        assert_eq!(rewrite("Co-op stores sell food."), "Co-op stores sell food.");
        assert_eq!(rewrite("Coca-Cola Co is a brand."), "Coca-Cola Co is a brand.");
        assert_eq!(rewrite("We ate at the co. canteen."), "We ate at the co. canteen.");
    }

    #[test]
    fn repeated_and_combined_marks() {
        assert_eq!(rewrite("Yes!! Let's go."), "Yes! Let's go.");
        assert_eq!(rewrite("Wow!!! That is great!!"), "Wow! That is great!");
        assert_eq!(rewrite("What?! Really?"), "What? Really?");
        assert_eq!(rewrite("Really!? Yes!?"), "Really? Yes?");
        assert_eq!(rewrite("Well... maybe later."), "Well, maybe later.");
        assert_eq!(rewrite("The sale ends soon... hurry!"), "The sale ends soon, hurry!");
        assert_eq!(rewrite("The sale ends soon...hurry!"), "The sale ends soon, hurry!");
        assert_eq!(rewrite("It is over..."), "It is over.");
        assert_eq!(rewrite("Wait\u{2026}"), "Wait.");
        assert_eq!(rewrite("Hello!"), "Hello!");
        assert_eq!(rewrite("1...2"), "1...2");
        assert_eq!(rewrite("Wait.... no"), "Wait.... no");
    }

    #[test]
    fn empty_long_and_unusual_text_is_safe() {
        assert_eq!(rewrite(""), "");
        assert_eq!(rewrite("   "), "   ");
        assert_eq!(rewrite("Dr."), "Dr.");
        assert_eq!(rewrite("Ave."), "Ave.");
        assert_eq!(rewrite("Maple Dr"), "Maple Dr");
        assert_eq!(rewrite("Elm Ave.\tnear the park"), "Elm Avenue\tnear the park");
        assert_eq!(rewrite("Café Ave. near the 漢字 park"), "Café Avenue near the 漢字 park");
        assert_eq!(rewrite("naïve… really"), "naïve, really");
        assert_eq!(rewrite("…"), "");
        assert_eq!(punctuation("!!!"), "!");
        let dots = "Mt. ".repeat(20_000);
        assert!(rewrite(&dots).starts_with("Mount Mount Mount"));
        let long = "Maple Dr. ".repeat(20_000);
        let written = rewrite(&long);
        assert!(written.starts_with("Maple Dr. Maple Dr. "));
        assert!(written.ends_with("Maple Drive. "));
        let elms = "Elm Ave. ".repeat(20_000);
        assert!(rewrite(&elms).starts_with("Elm Avenue. Elm Avenue."));
        let word = "a".repeat(100_000);
        assert_eq!(rewrite(&word), word);
    }
}
