//! Web addresses, e-mail addresses and symbols, written as the words a person would say. "labzero.ai" is read as
//! "lab zero dot A I", "example.com/page" as "example dot com slash page" and "name@example.com" as "name at example dot
//! com". A sign on its own or between values is written as its word: "=" is "equals", "<" is "less than", "#1" is
//! "number 1", "~20" is "about 20", "A+" is "A plus" and "yes/no" is "yes or no". A sign with no spoken form ("*" on its
//! own, "\u{ae}", "\u{2122}") is dropped. A line written only in capitals, whose capital words are all common words, is
//! read in sentence case.

/// Top-level domains that make a bare word such as "example.com" a web address. Other endings, such as "report.pdf" and
/// "3.14", are left alone.
const TOP_LEVEL_DOMAINS: &[&str] = &["com", "org", "net", "edu", "gov", "mil", "info", "io", "ai", "co", "uk", "dev", "app"];

/// Domain labels with a spoken form of their own: "www" is three letters, "ai" and "io" are letters, and the lab's name is
/// two words.
const LABEL_WORDS: &[(&str, &str)] = &[("www", "w w w"), ("labzero", "lab zero"), ("ai", "A I"), ("io", "I O")];

/// Common words of two to five letters, for lines written only in capitals. Initialisms and words that often stand for
/// initialisms are left out ("US", "IT", "AM", "PM", "CAT", "PIN"), and so are longer words.
const COMMON_CAPITALS: &str = "\
an as at be by do go he hi if in is me my no of oh on or so to up we \
add age aid all and any ask bad bag bed big bit box boy bus buy can car cut day did die dog dry due eat end far few fit fly \
for fun gas get got had has her him his hot how its job key kid let lie lot low man map may men met mix mom net new not now \
odd off old one our out own pay pen put raw red run sad saw say see set she sit six sky son sun tax tea ten the too top toy \
try two use war way wet who why win won yes yet you zoo paid \
able also area away back ball bank base best bike blue boat body book both bill call came card care case cash city come \
cook cool dark deal dear deep does done door down draw drop each earn easy even ever face fact fair fall fast fear feel feet \
file fill find fine fire firm fish five flat flow food foot form four free from full game gave gift girl give glad goes gold \
gone good grow hair half hall hand hard have head hear heat help here hide high hold home hope hour idea into iron item join \
jump just keep kind know land last late lead left less life lift like line list live long look loss lose lost loud love made \
mail main make many mark meal mean meet mind mine miss more most move much must name near need next nice nine none note once \
only open over page pain pair park part pass past pick plan play plus poor post pull push race rain rate read real rest rich \
ride ring rise road rock rule rush safe sale salt same sand save seat seed send sent show side sign site size slow snow soft \
some song soon sort spot star stay step stop such sure take talk tall team tell test text than that them then they thin this \
time tiny told took tool town tree trip true turn type used very view wait walk wall want warm wash week well went were what \
when wide will wind wish with wood word work year your zero \
about above after again alone along apply arms basic beach black block board break bring build carry catch cause chain \
chair check child claim class clean clear climb clock close cloud count cover cross daily dance death delay doubt draft \
dream dress drink drive early earth eight email enter equal error event every exact extra false fault field fight final \
first flame floor focus force found frame fresh front fruit funny glass going grade grand grant grass great green group \
guard guess guest guide happy heart heavy hello house human image issue judge juice known label large later laugh learn \
leave legal level light limit local logic lucky lunch major march match maybe media metal might minus model money month \
motor mouse mouth movie music never night noise north ocean offer often order other paint panel paper party phone photo \
piece place plain plane plant plate point power press price print prize proud quick quiet radio raise range ratio ready \
right river robot round route scale score sense serve seven shape share sharp sheet shift shirt shoot short since sleep \
slide small smart smile solid solve sorry sound south space speak speed spend split sport staff stand start state stick \
still stock stone store storm story study style sugar sweet table thank their theme thing think third those three throw \
tight title today topic total touch tower track trade train treat trial truck trust truth twice under union until upper \
video visit voice waste watch water wheel where which while white whole whose woman world worry worse worst worth would \
write wrong young youth";

/// The text with addresses and symbols written as words.
pub(super) fn rewrite(text: &str) -> String {
    let (pieces, trailing) = split_tokens(text);
    let mut out = String::with_capacity(text.len() + 16);
    if let Some(&(leading, _)) = pieces.first() {
        out.push_str(leading);
    }
    let mut emitted = false;
    let equations = equation_ahead(&pieces);
    for (i, &(space, token)) in pieces.iter().enumerate() {
        let prev = i.checked_sub(1).map(|j| pieces[j].1);
        let next = pieces.get(i + 1).map(|piece| piece.1);
        let spoken = rewrite_token(token, prev, next, equations[i]).unwrap_or_else(|| token.to_string());
        if spoken.is_empty() {
            continue;
        }
        if emitted {
            out.push_str(space);
        }
        out.push_str(&spoken);
        emitted = true;
    }
    out.push_str(trailing);
    if is_capitals_line(text) {
        match lower_capital_words(&out) {
            Some(lowered) => lowered,
            None => out,
        }
    } else {
        out
    }
}

/// Splits text into whitespace-separated tokens. Each token comes with the whitespace before it; the whitespace after the
/// last token is returned on its own.
fn split_tokens(text: &str) -> (Vec<(&str, &str)>, &str) {
    let mut pieces = Vec::new();
    let mut rest = text;
    loop {
        let body = rest.trim_start();
        let space = &rest[..rest.len() - body.len()];
        if body.is_empty() {
            return (pieces, space);
        }
        let end = body.find(char::is_whitespace).unwrap_or(body.len());
        pieces.push((space, &body[..end]));
        rest = &body[end..];
    }
}

/// The spoken form of one token, or None when the token stays as it is. An empty result drops the token.
fn rewrite_token(token: &str, prev: Option<&str>, next: Option<&str>, equation: bool) -> Option<String> {
    let (lead, core, trail) = split_edges(token);
    let before = prev.and_then(|p| split_edges(p).1.chars().last());
    let after = next.and_then(|n| split_edges(n).1.chars().next());
    let core = rewrite_core(core, before, after, equation)?;
    Some(format!("{lead}{core}{trail}"))
}

/// For each token, whether an "=" follows it in the same sentence. The token that ends the sentence still counts. The
/// scan runs backwards, so the whole line takes linear time.
fn equation_ahead(pieces: &[(&str, &str)]) -> Vec<bool> {
    let mut ahead = vec![false; pieces.len()];
    let mut found = false;
    for (i, &(_, token)) in pieces.iter().enumerate().rev() {
        ahead[i] = found;
        found = split_edges(token).1.contains('=') || (!ends_sentence(token) && found);
    }
    ahead
}

/// True when a token ends a sentence: a full stop, question mark or exclamation mark, possibly followed by closing brackets
/// or quotes.
fn ends_sentence(token: &str) -> bool {
    token
        .trim_end_matches(|c: char| matches!(c, ')' | ']' | '}' | '"' | '\'' | '\u{201d}' | '\u{2019}'))
        .ends_with(['.', '!', '?'])
}

fn is_opening(c: char) -> bool {
    matches!(c, '(' | '[' | '{' | '"' | '\'' | '\u{201c}' | '\u{2018}')
}

fn is_closing(c: char) -> bool {
    matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '"' | '\'' | '\u{201d}' | '\u{2019}')
}

/// Splits a token into its leading brackets and quotes, its core and its trailing punctuation. A token that is only
/// punctuation, such as ":", is all core.
fn split_edges(token: &str) -> (&str, &str, &str) {
    let body = token.trim_start_matches(is_opening);
    let lead = &token[..token.len() - body.len()];
    let core = body.trim_end_matches(is_closing);
    if core.is_empty() {
        return (lead, body, "");
    }
    (lead, core, &body[core.len()..])
}

/// The spoken form of the core of a token. `before` and `after` are the characters next to the token, when they are in
/// the neighbouring tokens.
fn rewrite_core(core: &str, before: Option<char>, after: Option<char>, equation: bool) -> Option<String> {
    if core.eq_ignore_ascii_case("yes/no") {
        return Some(core.replacen('/', " or ", 1));
    }
    if let Some(spoken) = spoken_address(core) {
        return Some(spoken);
    }
    if looks_like_address(core) {
        return None;
    }
    if let Some(word) = standalone_sign(core, before, after, equation) {
        return Some(word.to_string());
    }
    if let Some(spoken) = prefixed_sign(core).or_else(|| plus_suffix(core)) {
        return Some(spoken);
    }
    let changed = infix_signs(core);
    (changed != core).then_some(changed)
}

/// A web or e-mail address with a known top-level domain, spoken as words. None when the text is not such an address.
fn spoken_address(core: &str) -> Option<String> {
    let spoken = if let Some((local, domain)) = core.split_once('@') {
        format!("{} at {}", spoken_local(local)?, spoken_domain(domain, false)?)
    } else {
        let address = strip_scheme(core);
        let (host, path) = match address.find('/') {
            Some(at) => address.split_at(at),
            None => (address, ""),
        };
        let domain = spoken_domain(host, true)?;
        if path.is_empty() {
            domain
        } else {
            format!("{domain}{}", spoken_path(path)?)
        }
    };
    Some(spoken.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Something that is meant as an address but cannot be read as one (an "@", a scheme, or a path) stays as it is.
fn looks_like_address(core: &str) -> bool {
    core.contains('@') || core.contains("://") || (core.contains('/') && core.contains('.'))
}

/// The address without an "http://" or "https://" scheme, which is not spoken.
fn strip_scheme(core: &str) -> &str {
    for scheme in ["https://", "http://"] {
        let bytes = core.as_bytes();
        if bytes.len() >= scheme.len() && bytes[..scheme.len()].eq_ignore_ascii_case(scheme.as_bytes()) {
            return &core[scheme.len()..];
        }
    }
    core
}

/// The local part of an e-mail address: a dot is "dot" and an underscore is "underscore".
fn spoken_local(local: &str) -> Option<String> {
    let plain = !local.is_empty()
        && !local.starts_with('.')
        && !local.ends_with('.')
        && !local.contains("..")
        && local.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+'));
    plain.then(|| local.replace('.', " dot ").replace('_', " underscore "))
}

/// A domain such as "docs.python.org": each label is spoken, and the labels are joined with "dot". A bare word needs a
/// top-level domain from the list; an e-mail domain may end in any letters.
fn spoken_domain(host: &str, known_top_level_only: bool) -> Option<String> {
    let labels: Vec<&str> = host.split('.').collect();
    if labels.len() < 2 || !labels.iter().all(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')) {
        return None;
    }
    let top = labels[labels.len() - 1].to_ascii_lowercase();
    let top_ok = if known_top_level_only {
        TOP_LEVEL_DOMAINS.contains(&top.as_str())
    } else {
        top.len() >= 2 && top.chars().all(|c| c.is_ascii_alphabetic())
    };
    if !top_ok {
        return None;
    }
    let spoken: Vec<String> = labels.iter().map(|label| spoken_label(label)).collect();
    Some(spoken.join(" dot "))
}

fn spoken_label(label: &str) -> String {
    let lower = label.to_ascii_lowercase();
    for &(from, to) in LABEL_WORDS {
        if lower == from {
            return to.to_string();
        }
    }
    lower
}

/// A path after the domain: each "/" is "slash" and each dot is "dot". Other characters make the address stay as it is.
fn spoken_path(path: &str) -> Option<String> {
    let body = path.strip_prefix('/')?;
    let segments: Vec<&str> = body.split('/').collect();
    let last = segments.len() - 1;
    let mut out = String::new();
    for (i, segment) in segments.iter().enumerate() {
        let plain = segment.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
        if !plain || (segment.is_empty() && i != last) {
            return None;
        }
        out.push_str(" slash ");
        out.push_str(&segment.replace('.', " dot ").replace('_', " underscore "));
    }
    Some(out)
}

/// A sign that stands on its own, or that has values on both sides, as a word. An empty word drops the sign. A spaced
/// hyphen between numbers is a minus sign only in an equation, that is when an "=" follows in the same sentence.
fn standalone_sign(core: &str, before: Option<char>, after: Option<char>, equation: bool) -> Option<&'static str> {
    let between_values = before.is_some_and(char::is_alphanumeric) && after.is_some_and(char::is_alphanumeric);
    let between_numbers = before.is_some_and(|c| c.is_ascii_digit()) && after.is_some_and(|c| c.is_ascii_digit());
    let next_digit = after.is_some_and(|c| c.is_ascii_digit());
    match core {
        "=" if between_values => Some("equals"),
        "<" if between_values => Some("less than"),
        ">" if between_values => Some("greater than"),
        "-" if between_numbers && equation => Some("minus"),
        ":" if between_numbers => Some("to"),
        "*" if between_numbers => Some("times"),
        "*" => Some(""),
        "~" if next_digit => Some("about"),
        "#" if next_digit => Some("number"),
        _ => None,
    }
}

/// "#1" is "number 1", "#launch" is "hashtag launch" and "~20" is "about 20".
fn prefixed_sign(core: &str) -> Option<String> {
    if let Some(rest) = core.strip_prefix('#') {
        let first = rest.chars().next()?;
        let word = if first.is_ascii_digit() {
            "number"
        } else if first.is_ascii_alphabetic() {
            "hashtag"
        } else {
            return None;
        };
        return Some(format!("{word} {rest}"));
    }
    let rest = core.strip_prefix('~')?;
    rest.starts_with(|c: char| c.is_ascii_digit()).then(|| format!("about {rest}"))
}

/// A plus sign after a letter or a digit is "plus", as in "A+" and "C++".
fn plus_suffix(core: &str) -> Option<String> {
    let base = core.trim_end_matches('+');
    let pluses = core.len() - base.len();
    if pluses == 0 || base.is_empty() || !base.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(format!("{base}{}", " plus".repeat(pluses)))
}

/// Signs inside a token: "x<5", "Q&A", "5±0.2", "A→B" and "©2026" are written as words, "*" is dropped unless it is
/// between digits, and "™" and "®" are dropped.
fn infix_signs(core: &str) -> String {
    let chars: Vec<char> = core.chars().collect();
    let mut out = String::with_capacity(core.len() + 16);
    for (i, &c) in chars.iter().enumerate() {
        let before = i.checked_sub(1).map(|j| chars[j]);
        let after = chars.get(i + 1).copied();
        let alnum_before = before.is_some_and(char::is_alphanumeric);
        let alnum_after = after.is_some_and(char::is_alphanumeric);
        let digit_after = after.is_some_and(|d| d.is_ascii_digit());
        let digits_around = before.is_some_and(|d| d.is_ascii_digit()) && digit_after;
        match c {
            '=' if alnum_before && alnum_after => out.push_str(" equals "),
            '&' if alnum_before && alnum_after => out.push_str(" and "),
            '<' if alnum_before && digit_after => out.push_str(" less than "),
            '>' if alnum_before && digit_after => out.push_str(" greater than "),
            '*' if digits_around => out.push_str(" times "),
            '*' | '\u{ae}' | '\u{2122}' => {}
            '\u{b1}' => out.push_str(" plus or minus "),
            '\u{2192}' => out.push_str(" to "),
            '\u{a9}' => out.push_str(" copyright "),
            _ => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True when the line has capital letters and no lower-case letters or digits.
fn is_capitals_line(text: &str) -> bool {
    let mut has_capital = false;
    for c in text.chars() {
        if c.is_lowercase() || c.is_ascii_digit() {
            return false;
        }
        has_capital |= c.is_uppercase();
    }
    has_capital
}

/// On a line written only in capitals, every capital word of two or more letters must be in `COMMON_CAPITALS`. Then the
/// words are lower-cased and each sentence starts with a capital. Otherwise the line is returned as None and stays as it
/// is, because a capital word left among lower-case ones is spelled out letter by letter.
fn lower_capital_words(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_alphabetic() {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && chars[i].is_ascii_alphabetic() {
            i += 1;
        }
        let run: String = chars[start..i].iter().collect();
        if run.len() < 2 || !run.chars().all(|c| c.is_ascii_uppercase()) {
            out.push_str(&run);
            continue;
        }
        let joined = (start > 0 && is_word_joiner(chars[start - 1])) || (i < chars.len() && is_word_joiner(chars[i]));
        let lower = run.to_ascii_lowercase();
        if joined || !COMMON_CAPITALS.split_whitespace().any(|w| w == lower) {
            return None;
        }
        out.push_str(&lower);
    }
    Some(capitalise_sentences(&out))
}

/// Capitalises the first letter of the text and of each sentence after a full stop, question mark or exclamation mark.
fn capitalise_sentences(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut sentence_start = true;
    let mut after_stop = false;
    for c in text.chars() {
        if c.is_alphabetic() {
            if sentence_start {
                out.extend(c.to_uppercase());
            } else {
                out.push(c);
            }
            sentence_start = false;
            after_stop = false;
        } else {
            out.push(c);
            if matches!(c, '.' | '!' | '?') {
                after_stop = true;
            } else if c.is_whitespace() {
                sentence_start |= after_stop;
            } else {
                after_stop = false;
            }
        }
    }
    out
}

fn is_word_joiner(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '\'' || c == '-'
}

#[cfg(test)]
mod tests {
    use super::rewrite;

    fn check(input: &str, expected: &str) {
        assert_eq!(rewrite(input), expected, "input: {input:?}");
    }

    #[test]
    fn bare_domains_are_spoken_with_letters_for_ai_and_io() {
        check("Visit labzero.ai for details.", "Visit lab zero dot A I for details.");
        check("Our page is site.io/about.", "Our page is site dot I O slash about.");
        check("The docs live at docs.python.org.", "The docs live at docs dot python dot org.");
    }

    #[test]
    fn paths_schemes_and_www_are_spoken() {
        check("See example.com/page for more.", "See example dot com slash page for more.");
        check("The file is at example.com/a/b.html.", "The file is at example dot com slash a slash b dot html.");
        check("Go to https://www.example.org today.", "Go to w w w dot example dot org today.");
        check("Visit labzero.ai.", "Visit lab zero dot A I.");
    }

    #[test]
    fn e_mail_addresses_are_spoken_with_at() {
        check("Email name@example.com for help.", "Email name at example dot com for help.");
        check("Mail me at jane.doe@example.co.uk.", "Mail me at jane dot doe at example dot co dot uk.");
        check("Send it to info@labzero.ai.", "Send it to info at lab zero dot A I.");
        check("Write to first_last@example.net.", "Write to first underscore last at example dot net.");
    }

    #[test]
    fn file_names_versions_decimals_and_titles_are_kept() {
        for same in [
            "Open report.pdf and index.html today.",
            "The version is v1.2.3 and pi is 3.14.",
            "I live in the U.S.",
            "Dr. Smith will see you at noon.",
            "The e-mail was sent.",
            "Node.js and the README.md file.",
        ] {
            check(same, same);
        }
    }

    #[test]
    fn signs_on_their_own_or_between_values_are_words() {
        check("If x < 5, stop.", "If x less than 5, stop.");
        check("If y > 10, continue.", "If y greater than 10, continue.");
        check("Solve 3 + 5 = x.", "Solve 3 + 5 equals x.");
        check("The result is 5 \u{b1} 0.2.", "The result is 5 plus or minus 0.2.");
        check("Accuracy is \u{b1}3 meters.", "Accuracy is plus or minus 3 meters.");
        check("She was the #1 seed.", "She was the number 1 seed.");
        check("Show the #launch post.", "Show the hashtag launch post.");
        check("We expect ~20 people.", "We expect about 20 people.");
        check("Route A \u{2192} B.", "Route A to B.");
        check("Your grade was an A+.", "Your grade was an A plus.");
        check("Answer yes/no on the form.", "Answer yes or no on the form.");
        check("The ratio is 3 : 4.", "The ratio is 3 to 4.");
    }

    #[test]
    fn spaced_hyphens_between_numbers_are_minus_only_in_equations() {
        check("Subtract ten: 10 - 3 = 7.", "Subtract ten: 10 minus 3 equals 7.");
        check("Pages 5 - 10 are missing.", "Pages 5 - 10 are missing.");
        check("Score: 3 - 2.", "Score: 3 - 2.");
        check("The answer is 5 - 2.", "The answer is 5 - 2.");
        check("Compute 10 - 3. Then x = 5.", "Compute 10 - 3. Then x equals 5.");
    }

    #[test]
    fn arithmetic_signs_and_ampersands_are_words() {
        check("Compute 10 - 3 = 7 and 3 * 4.", "Compute 10 minus 3 equals 7 and 3 times 4.");
        check("Compute 10 - 3 and 3 * 4.", "Compute 10 - 3 and 3 times 4.");
        check("Q&A starts at noon.", "Q and A starts at noon.");
        check("Tom & Jerry is a classic show.", "Tom & Jerry is a classic show.");
        check("Lewis&Clark explored the west.", "Lewis and Clark explored the west.");
    }

    #[test]
    fn copyright_trademark_and_star_signs() {
        check("\u{a9} 2026 Acme printing press.", "copyright 2026 Acme printing press.");
        check("Acme\u{2122} and Brand\u{ae} tools.", "Acme and Brand tools.");
        check("Use the * symbol here.", "Use the symbol here.");
        check("Make it **bold** now.", "Make it bold now.");
    }

    #[test]
    fn signs_of_other_forms_and_unclear_cases_are_kept() {
        check("Compute 3 \u{d7} 4 = 12.", "Compute 3 \u{d7} 4 equals 12.");
        check("C# and F# are languages.", "C# and F# are languages.");
        check("Use <b>bold</b> text.", "Use <b>bold</b> text.");
        check("Mix 10-3 and 3<=4.", "Mix 10-3 and 3<=4.");
        check("Visit example.com/?q=1 for more.", "Visit example.com/?q=1 for more.");
    }

    #[test]
    fn all_capital_lines_of_common_words_read_as_sentences() {
        check("THE END IS NEAR.", "The end is near.");
        check("DO NOT ENTER.", "Do not enter.");
        check("THIS IS LOUD.", "This is loud.");
        check("NOW OR NEVER.", "Now or never.");
        check("I LOVE YOU.", "I love you.");
        check("STOP. GO NOW.", "Stop. Go now.");
    }

    #[test]
    fn all_capital_lines_with_other_capital_words_are_kept() {
        check("WARNING: HOT SURFACE.", "WARNING: HOT SURFACE.");
        check("NASA AND US", "NASA AND US");
        check("PLEASE READ THE LABEL.", "PLEASE READ THE LABEL.");
        check("NEW YORK IS BIG.", "NEW YORK IS BIG.");
    }

    #[test]
    fn mixed_or_numbered_lines_keep_their_capitals() {
        check("We won the GAME.", "We won the GAME.");
        check("NASA says IT works.", "NASA says IT works.");
        check("STAY 2 DAYS AT HOME.", "STAY 2 DAYS AT HOME.");
    }

    #[test]
    fn empty_blank_and_unicode_text_is_safe() {
        check("", "");
        check("   ", "   ");
        check(
            "caf\u{e9} \u{2122} na\u{ef}ve@example.com \u{65e5}\u{672c} #1 ~5",
            "caf\u{e9} na\u{ef}ve@example.com \u{65e5}\u{672c} number 1 about 5",
        );
        check("\u{2014}\u{2014} \u{1f600} \u{e9}.com", "\u{2014}\u{2014} \u{1f600} \u{e9}.com");
        check("x =", "x =");
        check("http://", "http://");
        check("@", "@");
        check("~", "~");
        check("#", "#");
    }

    #[test]
    fn long_text_is_read_in_full() {
        let mut text = "Visit labzero.ai now. ".repeat(2000);
        text.push_str("Then x < 5.");
        let out = rewrite(&text);
        assert!(out.ends_with("Then x less than 5."), "end of output: {:?}", &out[out.len().saturating_sub(40)..]);
        assert_eq!(out.matches("lab zero dot A I").count(), 2000);
    }

    #[test]
    fn any_pair_of_signs_and_letters_is_safe() {
        let set: Vec<char> = "@#=<>+~\u{b1}\u{a9}\u{ae}\u{2122}*/&\u{2192}:.-_()[]{}\"'?!%$ \t\u{e9}\u{65e5}\u{1f600}\u{0}aZ1 9"
            .chars()
            .collect();
        for &a in &set {
            for &b in &set {
                let texts = [
                    format!("{a}{b}"),
                    format!("x {a} {b} y."),
                    format!("{a}x{b}1"),
                    format!("A{a}B{b}C"),
                    format!("{a} ABC {b}"),
                    format!("THE {a}{b} END"),
                ];
                for text in &texts {
                    let _ = rewrite(text);
                }
            }
        }
    }

    #[test]
    fn common_capital_words_are_short_lower_case_words() {
        for word in super::COMMON_CAPITALS.split_whitespace() {
            assert!((2..=5).contains(&word.len()), "{word}");
            assert!(word.chars().all(|c| c.is_ascii_lowercase()), "{word}");
        }
        for banned in ["us", "it", "am", "pm", "cat", "pin"] {
            assert!(!super::COMMON_CAPITALS.split_whitespace().any(|w| w == banned), "{banned}");
        }
    }
}
