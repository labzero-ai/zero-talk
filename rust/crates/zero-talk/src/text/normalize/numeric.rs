//! Numbers with units and structure: units after a number, ranges and ratios, phone numbers and codes, ISO and written dates,
//! money with a scale, degrees, clock times, fractions, Roman numerals, superscripts and multiplication signs.
//! Digits stay digits unless the engine reads them differently from the spelled form: a decimal is spelled ("point" is
//! read stressed only in words), and so is a compound from 21 to 99 that does not end in zero, which is written with a
//! hyphen ("fifty-five").

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES: [(u64, &str); 4] = [
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];
/// Month names with their abbreviation.
const MONTHS: &[(&str, &str)] = &[
    ("January", "Jan"),
    ("February", "Feb"),
    ("March", "Mar"),
    ("April", "Apr"),
    ("May", "May"),
    ("June", "Jun"),
    ("July", "Jul"),
    ("August", "Aug"),
    ("September", "Sep"),
    ("October", "Oct"),
    ("November", "Nov"),
    ("December", "Dec"),
];
/// Unit tokens after a number: (token, singular, plural).
const UNITS: &[(&str, &str, &str)] = &[
    ("km", "kilometer", "kilometers"),
    ("m", "meter", "meters"),
    ("cm", "centimeter", "centimeters"),
    ("mm", "millimeter", "millimeters"),
    ("mi", "mile", "miles"),
    ("yd", "yard", "yards"),
    ("ft", "foot", "feet"),
    ("lb", "pound", "pounds"),
    ("lbs", "pound", "pounds"),
    ("oz", "ounce", "ounces"),
    ("kg", "kilogram", "kilograms"),
    ("g", "gram", "grams"),
    ("mg", "milligram", "milligrams"),
    ("L", "liter", "liters"),
    ("mL", "milliliter", "milliliters"),
    ("ml", "milliliter", "milliliters"),
    ("gal", "gallon", "gallons"),
    ("qt", "quart", "quarts"),
    ("KB", "kilobyte", "kilobytes"),
    ("MB", "megabyte", "megabytes"),
    ("GB", "gigabyte", "gigabytes"),
    ("TB", "terabyte", "terabytes"),
    ("Kbps", "kilobit per second", "kilobits per second"),
    ("Mbps", "megabit per second", "megabits per second"),
    ("Gbps", "gigabit per second", "gigabits per second"),
    ("MB/s", "megabyte per second", "megabytes per second"),
    ("GB/s", "gigabyte per second", "gigabytes per second"),
    ("Hz", "hertz", "hertz"),
    ("kHz", "kilohertz", "kilohertz"),
    ("MHz", "megahertz", "megahertz"),
    ("GHz", "gigahertz", "gigahertz"),
    ("V", "volt", "volts"),
    ("W", "watt", "watts"),
    ("kW", "kilowatt", "kilowatts"),
    ("MW", "megawatt", "megawatts"),
    ("kWh", "kilowatt hour", "kilowatt hours"),
    ("dB", "decibel", "decibels"),
    ("ppm", "part per million", "parts per million"),
    ("rpm", "revolution per minute", "revolutions per minute"),
    ("mph", "mile per hour", "miles per hour"),
    ("km/h", "kilometer per hour", "kilometers per hour"),
    ("kph", "kilometer per hour", "kilometers per hour"),
    ("m/s", "meter per second", "meters per second"),
    ("min", "minute", "minutes"),
    ("hr", "hour", "hours"),
    ("hrs", "hour", "hours"),
    ("sec", "second", "seconds"),
    ("secs", "second", "seconds"),
];
/// Units of length, which take "square" or "cubic" before the noun when followed by a superscript.
const LENGTH_UNITS: &[&str] = &["m", "km", "cm", "mm", "mi", "yd", "ft"];
const DENOMINATORS: &[u64] = &[2, 3, 4, 5, 6, 7, 8, 9, 10];
/// Words after which a Roman numeral is read as a cardinal ("Chapter IV", "World War II").
const CARDINAL_CUES: &[&str] = &[
    "Chapter", "Part", "Act", "Volume", "Vol", "Book", "Section", "Phase", "Level", "Type", "Table", "Bowl", "War", "Round",
    "Canto", "Scene", "Lesson", "Unit", "Episode", "Stage", "Series",
];
/// The cue words after which a single "I" is read as one; a pronoun "I" follows other words.
const CAPITAL_CUES: &[&str] = &["Part", "Chapter", "Act"];
/// Names after which a Roman numeral is read as a regnal or papal ordinal ("Henry VIII", "Pope John Paul II").
const MONARCHS: &[&str] = &[
    "Henry", "Elizabeth", "Louis", "Edward", "Mary", "Charles", "George", "James", "John", "Richard", "William", "Pius",
    "Paul", "Benedict", "Gregory", "Leo", "Innocent", "Clement", "Alexander", "Peter", "Catherine", "Philip", "Frederick",
    "Napoleon", "Ferdinand", "Ivan", "Otto", "Pope", "King", "Queen", "Tsar",
];
/// US state codes, which a five-digit ZIP code follows.
const STATES: &[&str] = &[
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
];

/// The text with those forms written as the words a person would say.
pub(super) fn rewrite(text: &str) -> String {
    let t = Text::new(text);
    let mut out = String::with_capacity(text.len() + 16);
    let mut i = 0;
    while i < t.c.len() {
        match rule_at(&t, i, &out) {
            Some((len, rep)) if len > 0 => {
                out.push_str(&rep);
                i += len;
            }
            _ => {
                out.push(t.c[i]);
                i += 1;
            }
        }
    }
    out
}

/// The text being rewritten, with flags read once from the whole segment.
struct Text {
    c: Vec<char>,
    /// For each position, whether an equals sign follows later in the same sentence: an arithmetic line keeps its dashes.
    eq_later: Vec<bool>,
    /// The segment names a ratio, so "N:M" is a ratio rather than a clock time.
    ratio: bool,
}

impl Text {
    fn new(s: &str) -> Text {
        let words: Vec<String> = s.split(|c: char| !c.is_alphabetic()).map(str::to_lowercase).collect();
        let has = |list: &[&str]| words.iter().any(|w| list.contains(&w.as_str()));
        let ratio = has(&["ratio", "ratios", "scale", "mix", "mixture", "aspect", "dilution", "proportion", "odds"]);
        let c: Vec<char> = s.chars().collect();
        let eq_later = equals_later(&c);
        Text { c, eq_later, ratio }
    }

    /// True when an equals sign follows position i later in the same sentence.
    fn equals_follows(&self, i: usize) -> bool {
        self.eq_later.get(i).copied().unwrap_or(false)
    }

    fn at(&self, i: usize) -> char {
        self.c.get(i).copied().unwrap_or('\0')
    }

    /// The number of ASCII digits from i.
    fn digits(&self, i: usize) -> usize {
        let mut j = i;
        while self.at(j).is_ascii_digit() {
            j += 1;
        }
        j - i
    }

    /// The value of len digits from i.
    fn num_at(&self, i: usize, len: usize) -> Option<u64> {
        (0..len).try_fold(0u64, |v, k| Some(v * 10 + u64::from(self.at(i + k).to_digit(10)?)))
    }

    /// The character before i does not join the number to a word, a sign or a decimal, so a number can start at i.
    fn word_start(&self, i: usize) -> bool {
        i == 0 || !joins(self.at(i - 1))
    }

    /// The position j ends a token: no word or slash follows, and no dash, point, colon or comma joins a digit.
    fn ends(&self, j: usize) -> bool {
        let ch = self.at(j);
        if word(ch) || ch == '/' {
            return false;
        }
        !(matches!(ch, '-' | '.' | ':' | ',') && self.at(j + 1).is_ascii_digit())
    }
}

/// For each position, whether an equals sign follows it later in the same sentence. A sentence ends at a full stop, a
/// question mark or an exclamation mark that is followed by white space or the end of the text.
fn equals_later(c: &[char]) -> Vec<bool> {
    let mut flags = vec![false; c.len()];
    let mut seen = false;
    for i in (0..c.len()).rev() {
        flags[i] = seen;
        match c[i] {
            '=' => seen = true,
            '.' | '!' | '?' if c.get(i + 1).map_or(true, |next| next.is_whitespace()) => seen = false,
            _ => {}
        }
    }
    flags
}

/// The first position at or after j that is not a space.
fn next_non_space(t: &Text, j: usize) -> usize {
    (j..).find(|&p| t.at(p) != ' ').unwrap_or(j)
}

fn word(ch: char) -> bool {
    ch.is_alphabetic() || ch.is_ascii_digit()
}

fn joins(ch: char) -> bool {
    word(ch) || matches!(ch, '.' | '/' | ':' | '-' | ',' | '$' | '£' | '€' | '@' | '#' | '_' | '°' | '²' | '³' | '%' | '&' | '+' | '=' | '×' | '–' | '\'')
}

fn rule_at(t: &Text, i: usize, out: &str) -> Option<(usize, String)> {
    if !t.word_start(i) {
        return None;
    }
    let ch = t.at(i);
    let rule = if ch.is_ascii_digit() {
        digit_rule(t, i, out)
    } else if matches!(ch, '$' | '£' | '€') {
        money_rule(t, i)
    } else if matches!(ch, '(' | '+') {
        phone_rule(t, i)
    } else if matches!(ch, '½' | '¼' | '¾') {
        unicode_fraction(t, i)
    } else if ch.is_ascii_uppercase() {
        roman_rule(t, i, out).or_else(|| month_day_rule(t, i)).or_else(|| superscript_rule(t, i))
    } else if ch.is_alphabetic() {
        superscript_rule(t, i)
    } else {
        None
    };
    rule.filter(|(len, _)| *len > 0)
}

fn digit_rule(t: &Text, i: usize, out: &str) -> Option<(usize, String)> {
    iso_date(t, i)
        .or_else(|| numeric_date(t, i))
        .or_else(|| phone_digits(t, i))
        .or_else(|| code_digits(t, i, out))
        .or_else(|| time_rule(t, i))
        .or_else(|| range_rule(t, i, out))
        .or_else(|| fraction_rule(t, i, out))
        .or_else(|| times_rule(t, i))
        .or_else(|| ordinal_of_month_rule(t, i))
        .or_else(|| day_month_rule(t, i))
        .or_else(|| number_suffix(t, i))
        .or_else(|| degree_rule(t, i, out))
        .or_else(|| unit_rule(t, i, out))
}

// ------------------------------------------------------------------------------------------------------ spelling

/// A number as written: its integer digits (commas removed), its decimal digits (trailing zeros removed) and the text as
/// written.
struct Num {
    int: String,
    frac: String,
    shown: String,
    end: usize,
}

impl Num {
    fn is_one(&self) -> bool {
        self.int == "1" && self.frac.is_empty()
    }
}

fn parse_num(t: &Text, i: usize) -> Option<Num> {
    if !t.at(i).is_ascii_digit() {
        return None;
    }
    let (mut j, mut int, mut shown) = (i, String::new(), String::new());
    loop {
        let ch = t.at(j);
        if ch.is_ascii_digit() {
            int.push(ch);
            shown.push(ch);
            j += 1;
        } else if ch == ',' && t.digits(j + 1) == 3 && !t.at(j + 4).is_ascii_digit() {
            for k in 1..=3 {
                int.push(t.at(j + k));
                shown.push(t.at(j + k));
            }
            shown.insert(shown.len() - 3, ',');
            j += 4;
        } else {
            break;
        }
    }
    let mut frac = String::new();
    if t.at(j) == '.' && t.at(j + 1).is_ascii_digit() {
        shown.push('.');
        j += 1;
        while t.at(j).is_ascii_digit() {
            frac.push(t.at(j));
            shown.push(t.at(j));
            j += 1;
        }
        while frac.ends_with('0') {
            frac.pop();
        }
    }
    Some(Num { int, frac, shown, end: j })
}

/// A plain run of digits as a number.
fn plain_num(digits: &str) -> Num {
    Num { int: digits.to_string(), frac: String::new(), shown: digits.to_string(), end: 0 }
}

/// True for 21 to 99 that do not end in zero: the compounds the engine reads with a space and the words write with a hyphen.
fn compound(v: u64) -> bool {
    let tail = v % 100;
    (21..=99).contains(&tail) && tail % 10 != 0
}

fn cardinal(n: u64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    for (size, name) in SCALES {
        if n >= size {
            let (top, rest) = (n / size, n % size);
            let mut s = format!("{} {}", cardinal(top), name);
            if rest > 0 {
                s.push(' ');
                s.push_str(&cardinal(rest));
            }
            return s;
        }
    }
    below_1000(n)
}

fn below_1000(n: u64) -> String {
    let (hundreds, rest) = (n / 100, n % 100);
    match (hundreds, rest) {
        (0, _) => below_100(rest),
        (_, 0) => format!("{} hundred", ONES[hundreds as usize]),
        _ => format!("{} hundred {}", ONES[hundreds as usize], below_100(rest)),
    }
}

fn below_100(n: u64) -> String {
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    let tens = TENS[(n / 10) as usize];
    if n % 10 == 0 {
        tens.to_string()
    } else {
        format!("{}-{}", tens, ONES[(n % 10) as usize])
    }
}

fn ordinal(n: u64) -> String {
    let words = cardinal(n);
    let cut = words.rfind(|c: char| c == ' ' || c == '-').map_or(0, |k| k + 1);
    let last = match &words[cut..] {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        w => match w.strip_suffix('y') {
            Some(stem) => format!("{stem}ieth"),
            None => format!("{w}th"),
        },
    };
    format!("{}{}", &words[..cut], last)
}

/// A day or other ordinal as it is read: compounds spelled, the rest as digits with their suffix.
fn ordinal_spoken(d: u64) -> String {
    if compound(d) {
        return ordinal(d);
    }
    let suffix = match (d % 100, d % 10) {
        (11..=13, _) => "th",
        (_, 1) => "st",
        (_, 2) => "nd",
        (_, 3) => "rd",
        _ => "th",
    };
    format!("{d}{suffix}")
}

/// A four-digit year as it is said: "twenty twenty-six", "nineteen oh five", "two thousand five".
fn year_words(y: u64) -> String {
    if y >= 1000 && y % 1000 < 10 {
        return cardinal(y);
    }
    let (high, low) = (y / 100, y % 100);
    if low == 0 {
        format!("{} hundred", cardinal(high))
    } else if low < 10 {
        format!("{} oh {}", cardinal(high), cardinal(low))
    } else {
        format!("{} {}", cardinal(high), below_100(low))
    }
}

fn digit_words(s: &str) -> String {
    s.chars().filter_map(|c| c.to_digit(10)).map(|d| ONES[d as usize]).collect::<Vec<_>>().join(" ")
}

/// A number as the engine should read it: a decimal or a compound spelled, anything else as written.
fn spoken(n: &Num) -> String {
    if !n.frac.is_empty() {
        let whole = match n.int.parse::<u64>() {
            Ok(v) if n.int.len() <= 15 => cardinal(v),
            _ => n.int.clone(),
        };
        return format!("{} point {}", whole, digit_words(&n.frac));
    }
    match n.int.parse::<u64>() {
        Ok(v) if n.int.len() <= 15 && compound(v) => cardinal(v),
        _ => n.shown.clone(),
    }
}

fn minutes_words(mm: u64) -> String {
    if mm < 10 {
        format!("oh {}", ONES[mm as usize])
    } else {
        cardinal(mm)
    }
}

fn month_name(m: u64) -> &'static str {
    MONTHS[(m - 1) as usize].0
}

/// The full name of a month written in full or as its usual abbreviation.
fn month_name_of(word: &str) -> Option<&'static str> {
    if word == "Sept" {
        return Some("September");
    }
    MONTHS.iter().find(|(full, abbr)| *full == word || *abbr == word).map(|(full, _)| *full)
}

/// A capitalised month word at j (with its full stop when it is an abbreviation): its full name and its length.
fn month_at(t: &Text, j: usize) -> Option<(&'static str, usize)> {
    let mut k = j;
    while k - j < 9 && t.at(k).is_ascii_alphabetic() {
        k += 1;
    }
    if k == j || word(t.at(k)) && !t.at(k).is_ascii_alphabetic() {
        return None;
    }
    let text: String = (j..k).map(|p| t.at(p)).collect();
    let name = month_name_of(&text)?;
    if text != name && t.at(k) == '.' {
        return Some((name, k + 1 - j));
    }
    if t.at(k).is_alphabetic() {
        return None;
    }
    Some((name, k - j))
}

/// " YYYY" after a day: its spelled year and its length.
fn year_suffix(t: &Text, end: usize) -> Option<(usize, String)> {
    if t.at(end) == ' ' && t.digits(end + 1) == 4 && t.ends(end + 5) {
        let y = t.num_at(end + 1, 4)?;
        return Some((5, year_words(y)));
    }
    None
}

/// The words before the current position, the last one first (at most four).
fn tail_words(out: &str) -> Vec<&str> {
    out.rsplit(char::is_whitespace).filter(|w| !w.is_empty()).take(4).collect()
}

/// True when the text before a number ends with the article "a" and a space.
fn article_before(out: &str) -> bool {
    let Some(rest) = out.strip_suffix(' ') else { return false };
    matches!(rest, "a" | "A") || rest.ends_with(" a") || rest.ends_with(" A")
}

/// True when the text before a number ends with a digit and a space (a mixed number such as "1 1/2").
fn mixed_number_before(out: &str) -> bool {
    out.strip_suffix(' ').is_some_and(|rest| rest.ends_with(|c: char| c.is_ascii_digit()))
}

fn roman_of(mut n: u64) -> String {
    const TABLE: [(u64, &str); 13] = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
        (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ];
    let mut s = String::new();
    for (value, numeral) in TABLE {
        while n >= value {
            s.push_str(numeral);
            n -= value;
        }
    }
    s
}

/// The value of a Roman numeral written in the canonical way, and None for anything else.
fn roman_value(s: &str) -> Option<u64> {
    let letter = |ch: char| -> Option<i64> {
        Some(match ch {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return None,
        })
    };
    let values = s.chars().map(letter).collect::<Option<Vec<i64>>>()?;
    let mut total = 0i64;
    for (k, v) in values.iter().enumerate() {
        if values.get(k + 1).is_some_and(|next| v < next) {
            total -= v;
        } else {
            total += v;
        }
    }
    if !(1..=3999).contains(&total) {
        return None;
    }
    let value = total as u64;
    (roman_of(value) == s).then_some(value)
}

fn capitalised(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

// ------------------------------------------------------------------------------------------------------ dates

fn iso_date(t: &Text, i: usize) -> Option<(usize, String)> {
    if t.digits(i) != 4 || t.at(i + 4) != '-' || t.digits(i + 5) != 2 || t.at(i + 7) != '-' || t.digits(i + 8) != 2 || !t.ends(i + 10) {
        return None;
    }
    let (y, m, d) = (t.num_at(i, 4)?, t.num_at(i + 5, 2)?, t.num_at(i + 8, 2)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some((10, format!("{} {} {}", month_name(m), ordinal_spoken(d), year_words(y))))
}

/// Numeric dates that only one reading allows: a field above 12 is the day.
fn numeric_date(t: &Text, i: usize) -> Option<(usize, String)> {
    let a = t.digits(i);
    if !(1..=2).contains(&a) || t.at(i + a) != '/' {
        return None;
    }
    let b_start = i + a + 1;
    let b = t.digits(b_start);
    if !(1..=2).contains(&b) || t.at(b_start + b) != '/' {
        return None;
    }
    let y_start = b_start + b + 1;
    if t.digits(y_start) != 4 || !t.ends(y_start + 4) {
        return None;
    }
    let (x, y, year) = (t.num_at(i, a)?, t.num_at(b_start, b)?, t.num_at(y_start, 4)?);
    let text = if (1..=12).contains(&x) && (13..=31).contains(&y) {
        format!("{} {} {}", month_name(x), ordinal_spoken(y), year_words(year))
    } else if (13..=31).contains(&x) && (1..=12).contains(&y) {
        format!("the {} of {} {}", ordinal_spoken(x), month_name(y), year_words(year))
    } else {
        return None;
    };
    Some((y_start + 4 - i, text))
}

/// "16th of July, 1969": an ordinal day, "of", a month and a year; the year is said in words.
fn ordinal_of_month_rule(t: &Text, i: usize) -> Option<(usize, String)> {
    let d = t.digits(i);
    if !(1..=2).contains(&d) {
        return None;
    }
    let suffix: String = (i + d..i + d + 2).map(|p| t.at(p)).collect();
    if !matches!(suffix.as_str(), "st" | "nd" | "rd" | "th") {
        return None;
    }
    let day = t.num_at(i, d)?;
    if !(1..=31).contains(&day) {
        return None;
    }
    let of_at = i + d + 2;
    if t.at(of_at) != ' ' || !is_of(t, of_at) || t.at(of_at + 3) != ' ' {
        return None;
    }
    let month_start = of_at + 4;
    let (_, mlen) = month_at(t, month_start)?;
    let month_end = month_start + mlen;
    let sep = if t.at(month_end) == ',' && t.at(month_end + 1) == ' ' {
        ", "
    } else if t.at(month_end) == ' ' {
        " "
    } else {
        return None;
    };
    let year_start = month_end + sep.len();
    if t.digits(year_start) != 4 || !t.ends(year_start + 4) {
        return None;
    }
    let year = t.num_at(year_start, 4)?;
    let month: String = (month_start..month_end).map(|p| t.at(p)).collect();
    Some((year_start + 4 - i, format!("{} of {}{}{}", ordinal_spoken(day), month, sep, year_words(year))))
}

/// "5 Nov 2026" and "12 December": the day, then the month, then an optional year.
fn day_month_rule(t: &Text, i: usize) -> Option<(usize, String)> {
    let d = t.digits(i);
    if !(1..=2).contains(&d) || t.at(i + d) != ' ' {
        return None;
    }
    let day = t.num_at(i, d)?;
    if !(1..=31).contains(&day) {
        return None;
    }
    let (month, mlen) = month_at(t, i + d + 1)?;
    let mut end = i + d + 1 + mlen;
    let mut text = format!("the {} of {}", ordinal_spoken(day), month);
    if let Some((len, year)) = year_suffix(t, end) {
        text.push(' ');
        text.push_str(&year);
        end += len;
    }
    Some((end - i, text))
}

/// "December 25", "Sept. 3rd": a capitalised month and a day in digits; the day is read as an ordinal unless a year follows it.
fn month_day_rule(t: &Text, i: usize) -> Option<(usize, String)> {
    let (_, mlen) = month_at(t, i)?;
    let j = i + mlen;
    if t.at(j) != ' ' {
        return None;
    }
    let d = t.digits(j + 1);
    if !(1..=2).contains(&d) {
        return None;
    }
    let day = t.num_at(j + 1, d)?;
    if !(1..=31).contains(&day) {
        return None;
    }
    let end = j + 1 + d;
    if !t.ends(end) {
        return None;
    }
    let after_comma = if t.at(end) == ',' { end + 1 } else { end };
    if t.at(next_non_space(t, after_comma)).is_ascii_digit() {
        return None;
    }
    let mut text: String = (i..j).map(|p| t.at(p)).collect();
    text.push(' ');
    text.push_str(&ordinal_spoken(day));
    Some((end - i, text))
}

// ------------------------------------------------------------------------------------------------------ codes and phones

fn spell_digits(t: &Text, start: usize, end: usize) -> String {
    digit_words(&(start..end).map(|p| t.at(p)).filter(char::is_ascii_digit).collect::<String>())
}

/// Phone numbers written 555-0134, 800-555-0199 and (555) 123-4567, and +1 555 123 4567 (from the plus sign).
fn phone_digits(t: &Text, i: usize) -> Option<(usize, String)> {
    if t.digits(i) != 3 || t.at(i + 3) != '-' {
        return None;
    }
    if t.digits(i + 4) == 3 && t.at(i + 7) == '-' && t.digits(i + 8) == 4 && t.ends(i + 12) {
        return Some((12, spell_digits(t, i, i + 12)));
    }
    if t.digits(i + 4) == 4 && t.ends(i + 8) {
        return Some((8, spell_digits(t, i, i + 8)));
    }
    None
}

fn phone_rule(t: &Text, i: usize) -> Option<(usize, String)> {
    if t.at(i) == '(' {
        let fits = t.digits(i + 1) == 3
            && t.at(i + 4) == ')'
            && t.at(i + 5) == ' '
            && t.digits(i + 6) == 3
            && t.at(i + 9) == '-'
            && t.digits(i + 10) == 4
            && t.ends(i + 14);
        return fits.then(|| (14, spell_digits(t, i, i + 14)));
    }
    let fits = t.at(i + 1) == '1'
        && t.at(i + 2) == ' '
        && t.digits(i + 3) == 3
        && t.at(i + 6) == ' '
        && t.digits(i + 7) == 3
        && t.at(i + 10) == ' '
        && t.digits(i + 11) == 4
        && t.ends(i + 15);
    fits.then(|| (15, format!("plus {}", spell_digits(t, i + 1, i + 15))))
}

/// Digit strings read digit by digit: leading-zero codes, ZIP codes, emergency numbers after "dial" or "call", PINs.
fn code_digits(t: &Text, i: usize, out: &str) -> Option<(usize, String)> {
    let n = t.digits(i);
    if n < 3 || !t.ends(i + n) {
        return None;
    }
    let words = tail_words(out);
    let last = words.first().copied().unwrap_or("");
    let digits: String = (i..i + n).map(|p| t.at(p)).collect();
    if t.at(i) == '0' {
        if n == 4 && last == "at" && digits.ends_with("00") {
            return Some((4, format!("oh {} hundred", ONES[t.at(i + 1).to_digit(10)? as usize])));
        }
        return Some((n, digit_words(&digits)));
    }
    let zip = n == 5 && (words.contains(&"ZIP") || STATES.contains(&last));
    let emergency = n == 3 && (digits == "911" || digits == "112") && matches!(last, "dial" | "Dial" | "call" | "Call");
    let pin = (4..=8).contains(&n) && words.iter().any(|w| matches!(*w, "PIN" | "pin" | "passcode"));
    if zip || emergency || pin {
        return Some((n, digit_words(&digits)));
    }
    None
}

// ------------------------------------------------------------------------------------------------------ times

/// " am", " a.m.", " PM" and so on after position j: the letters "A M" or "P M", and the length read.
fn ampm_at(t: &Text, j: usize) -> Option<(usize, &'static str)> {
    let k = if t.at(j) == ' ' { j + 1 } else { j };
    let ap = match t.at(k).to_ascii_lowercase() {
        'a' => "A M",
        'p' => "P M",
        _ => return None,
    };
    let mut m = k + 1;
    if t.at(m) == '.' {
        m += 1;
    }
    if t.at(m).to_ascii_lowercase() != 'm' {
        return None;
    }
    m += 1;
    let mut end = m;
    if t.at(m) == '.' && t.at(m + 1) == ' ' && t.at(m + 2).is_lowercase() {
        end = m + 1;
    }
    if word(t.at(end)) {
        return None;
    }
    Some((end - j, ap))
}

/// Clock times: "8:05" (eight oh five), "9:00 sharp" (nine o'clock), "10:00 am" (ten A M), "9 a.m." (nine A M).
fn time_rule(t: &Text, i: usize) -> Option<(usize, String)> {
    let h = t.digits(i);
    if !(1..=2).contains(&h) {
        return None;
    }
    let hour = t.num_at(i, h)?;
    let colon = i + h;
    if t.at(colon) == ':' && t.digits(colon + 1) == 2 {
        let mm = t.num_at(colon + 1, 2)?;
        let after = colon + 3;
        if mm >= 60 || t.at(after) == ':' {
            return None;
        }
        if let Some((len, ap)) = ampm_at(t, after) {
            if !(1..=12).contains(&hour) {
                return None;
            }
            let mins = if mm == 0 { String::new() } else { format!(" {}", minutes_words(mm)) };
            return Some((after + len - i, format!("{hour}{mins} {ap}")));
        }
        if !t.ends(after) {
            return None;
        }
        if mm == 0 {
            return Some((after - i, format!("{hour} o'clock")));
        }
        if mm < 10 || compound(mm) {
            return Some((after - i, format!("{} {}", hour, minutes_words(mm))));
        }
        return None;
    }
    let (len, ap) = ampm_at(t, colon)?;
    if !(1..=12).contains(&hour) {
        return None;
    }
    Some((h + len, format!("{hour} {ap}")))
}

// ------------------------------------------------------------------------------------------------------ ranges and fractions

/// The number a range side reads as: a compound spelled (a four-digit year as a year), anything else as digits.
fn range_part(digits: &str) -> String {
    let v: u64 = digits.parse().unwrap_or(0);
    if compound(v) {
        if digits.len() == 4 {
            year_words(v)
        } else {
            cardinal(v)
        }
    } else {
        digits.to_string()
    }
}

/// "5-10", "5–10", "5 - 10", "16:9" and "3 : 4": a range ("to") or a ratio. Dates, phone numbers and sums are left alone.
fn range_rule(t: &Text, i: usize, _out: &str) -> Option<(usize, String)> {
    let n_len = t.digits(i);
    if !(1..=4).contains(&n_len) || (t.at(i) == '0' && n_len > 1) {
        return None;
    }
    let n_end = i + n_len;
    let (kind, m_start) = match (t.at(n_end), t.at(n_end + 1), t.at(n_end + 2)) {
        ('-' | '–', _, _) => ('d', n_end + 1),
        (' ', '-' | '–', ' ') => ('s', n_end + 3),
        (':', _, _) => ('c', n_end + 1),
        (' ', ':', ' ') => ('C', n_end + 3),
        _ => return None,
    };
    let m_len = t.digits(m_start);
    if !(1..=4).contains(&m_len) || (t.at(m_start) == '0' && m_len > 1) {
        return None;
    }
    let m_end = m_start + m_len;
    if !t.ends(m_end) || t.at(m_end) == '=' {
        return None;
    }
    let ok = match kind {
        'd' => !(n_len == 3 && m_len == 4) && !(n_len == 4 && m_len == 2) && !t.equals_follows(n_end),
        's' => !t.equals_follows(n_end),
        'c' => (m_len != 2 || t.ratio) && !t.equals_follows(n_end),
        'C' => t.ratio && !t.equals_follows(n_end),
        _ => false,
    };
    if !ok {
        return None;
    }
    let n: String = (i..n_end).map(|p| t.at(p)).collect();
    let m: String = (m_start..m_end).map(|p| t.at(p)).collect();
    Some((m_end - i, format!("{} to {}", range_part(&n), range_part(&m))))
}

fn is_of(t: &Text, j: usize) -> bool {
    t.at(j) == ' ' && t.at(j + 1) == 'o' && t.at(j + 2) == 'f' && !word(t.at(j + 3))
}

/// The words for a fraction: "one half", "three quarter" before a noun, "two thirds of" with a plural before "of".
fn fraction_words(n: u64, d: u64, of: bool) -> String {
    let plural = n > 1 && of;
    let base = match d {
        2 => "half",
        3 => "third",
        4 => "quarter",
        5 => "fifth",
        6 => "sixth",
        7 => "seventh",
        8 => "eighth",
        9 => "ninth",
        10 => "tenth",
        11 => "eleventh",
        12 => "twelfth",
        _ => "sixteenth",
    };
    let denominator = match (plural, d) {
        (false, _) => base.to_string(),
        (true, 2) => "halves".to_string(),
        (true, _) => format!("{base}s"),
    };
    format!("{} {}", cardinal(n), denominator)
}

/// "1/2", "3/4", "5/8" in a quantity. A date ("1/2/2026") or a mixed number ("1 1/2") is left alone.
fn fraction_rule(t: &Text, i: usize, out: &str) -> Option<(usize, String)> {
    if t.at(i) == '0' || mixed_number_before(out) {
        return None;
    }
    let n_len = t.digits(i);
    if !(1..=2).contains(&n_len) || t.at(i + n_len) != '/' {
        return None;
    }
    let d_start = i + n_len + 1;
    let d_len = t.digits(d_start);
    if !(1..=2).contains(&d_len) {
        return None;
    }
    let end = d_start + d_len;
    if t.at(end) == '/' || !t.ends(end) {
        return None;
    }
    let (n, d) = (t.num_at(i, n_len)?, t.num_at(d_start, d_len)?);
    if n == 0 || n >= d || !DENOMINATORS.contains(&d) {
        return None;
    }
    Some((end - i, fraction_words(n, d, is_of(t, end))))
}

fn unicode_fraction(t: &Text, i: usize) -> Option<(usize, String)> {
    let (n, d) = match t.at(i) {
        '½' => (1, 2),
        '¼' => (1, 4),
        '¾' => (3, 4),
        _ => return None,
    };
    Some((1, fraction_words(n, d, is_of(t, i + 1))))
}

/// "3 x 4", "3x4" and "3 × 4" as "3 by 4" (or "3 times 4" before an equals sign); "2x faster" and "2 × faster" as "2 times faster".
fn times_rule(t: &Text, i: usize) -> Option<(usize, String)> {
    let n_len = t.digits(i);
    if !(1..=4).contains(&n_len) {
        return None;
    }
    let mut j = i + n_len;
    let spaced = t.at(j) == ' ';
    if spaced {
        j += 1;
    }
    let sign = t.at(j);
    if !matches!(sign, 'x' | '×') {
        return None;
    }
    j += 1;
    let n = plain_num(&(i..i + n_len).map(|p| t.at(p)).collect::<String>());
    let k = if t.at(j) == ' ' { j + 1 } else { j };
    let m_len = t.digits(k);
    if (1..=4).contains(&m_len) && t.ends(k + m_len) {
        let m = plain_num(&(k..k + m_len).map(|p| t.at(p)).collect::<String>());
        let word_between = if t.at(next_non_space(t, k + m_len)) == '=' { "times" } else { "by" };
        return Some((k + m_len - i, format!("{} {} {}", spoken(&n), word_between, spoken(&m))));
    }
    let multiplies = sign == '×' || !spaced;
    let next = next_non_space(t, j);
    if multiplies && !word(t.at(j)) && t.at(next).is_alphabetic() {
        return Some((j - i, format!("{} times", spoken(&n))));
    }
    None
}

// ------------------------------------------------------------------------------------------------------ units, money, degrees

fn unit_words(token: &str) -> Option<(&'static str, &'static str)> {
    UNITS.iter().find(|(tok, _, _)| *tok == token).map(|(_, sing, plur)| (*sing, *plur))
}

/// A unit after a number: "5 km" (five kilometers), "1 kWh" (one kilowatt hour), "a 2 TB drive" (a two terabyte drive),
/// "5 m²" (five square meters).
fn unit_rule(t: &Text, i: usize, out: &str) -> Option<(usize, String)> {
    let n = parse_num(t, i)?;
    let mut j = n.end;
    if t.at(j) == ' ' {
        j += 1;
    }
    let start = j;
    let mut k = j;
    while k - start < 8 && (t.at(k).is_ascii_alphabetic() || t.at(k) == '/') {
        k += 1;
    }
    if k == start {
        return None;
    }
    let token: String = (start..k).map(|p| t.at(p)).collect();
    let (sing, plur) = unit_words(&token)?;
    if word(t.at(k)) {
        return None;
    }
    if token == "m" && t.at(k) == '.' && matches!(t.at(k + 1), 'a' | 'p') {
        return None;
    }
    let one = n.is_one();
    let sup = t.at(k);
    if (sup == '²' || sup == '³') && LENGTH_UNITS.contains(&token.as_str()) {
        let prefix = if sup == '²' { "square" } else { "cubic" };
        let noun = if one { sing } else { plur };
        return Some((k + 1 - i, format!("{} {} {}", spoken(&n), prefix, noun)));
    }
    let noun = if one || article_before(out) { sing } else { plur };
    Some((k - i, format!("{} {}", spoken(&n), noun)))
}

/// "180°C" (degrees Celsius), "75°F", "20° north" (degrees), "a 90° turn" (a ninety degree turn).
fn degree_rule(t: &Text, i: usize, out: &str) -> Option<(usize, String)> {
    let n = parse_num(t, i)?;
    let mut j = n.end;
    if t.at(j) == ' ' {
        j += 1;
    }
    if t.at(j) != '°' {
        return None;
    }
    let k = j + 1;
    if t.at(k) == 'C' && !word(t.at(k + 1)) {
        return Some((k + 1 - i, format!("{} degrees Celsius", spoken(&n))));
    }
    if t.at(k) == 'F' && !word(t.at(k + 1)) {
        return Some((k + 1 - i, format!("{} degrees Fahrenheit", spoken(&n))));
    }
    if article_before(out) {
        return Some((k - i, format!("{} degree", spoken(&n))));
    }
    Some((k - i, format!("{} degrees", spoken(&n))))
}

/// "50¢", and "3.5k" (three point five thousand).
fn number_suffix(t: &Text, i: usize) -> Option<(usize, String)> {
    let n = parse_num(t, i)?;
    if t.at(n.end) == '¢' {
        let unit = if n.is_one() { "cent" } else { "cents" };
        return Some((n.end + 1 - i, format!("{} {}", spoken(&n), unit)));
    }
    if !n.frac.is_empty() && t.at(n.end) == 'k' && !word(t.at(n.end + 1)) {
        return Some((n.end + 1 - i, format!("{} thousand", spoken(&n))));
    }
    None
}

/// A scale after a money amount: "k", "M", "bn" attached, or " million" and so on spaced.
fn money_scale(t: &Text, j: usize) -> Option<(usize, &'static str)> {
    let attached = match t.at(j) {
        'k' | 'K' => Some((1, "thousand")),
        'm' | 'M' => Some((1, "million")),
        'b' | 'B' if matches!(t.at(j + 1), 'n' | 'N') => Some((2, "billion")),
        'b' | 'B' => Some((1, "billion")),
        _ => None,
    };
    if let Some((len, name)) = attached {
        if !word(t.at(j + len)) {
            return Some((len, name));
        }
    }
    if t.at(j) == ' ' {
        let mut k = j + 1;
        while k - j - 1 < 9 && t.at(k).is_ascii_alphabetic() {
            k += 1;
        }
        let text: String = (j + 1..k).map(|p| t.at(p)).collect();
        let name = ["thousand", "million", "billion", "trillion"].into_iter().find(|name| *name == text)?;
        if !word(t.at(k)) {
            return Some((k - j, name));
        }
    }
    None
}

/// "$2.5M" (two point five million dollars), "£2m", "€1.2bn", "$300k", "€20", "$5/month" (five dollars a month).
fn money_rule(t: &Text, i: usize) -> Option<(usize, String)> {
    let currency = match t.at(i) {
        '$' => "dollar",
        '£' => "pound",
        _ => "euro",
    };
    let n = parse_num(t, i + 1)?;
    if let Some((len, scale)) = money_scale(t, n.end) {
        return Some((n.end + len - i, format!("{} {} {}s", spoken(&n), scale, currency)));
    }
    if currency == "euro" {
        if !n.frac.is_empty() {
            return None;
        }
        let unit = if n.is_one() { "euro" } else { "euros" };
        return Some((n.end - i, format!("{} {}", spoken(&n), unit)));
    }
    if t.at(n.end) == '/' {
        let mut k = n.end + 1;
        while k - n.end - 1 < 8 && t.at(k).is_ascii_alphabetic() {
            k += 1;
        }
        let period = match (n.end + 1..k).map(|p| t.at(p)).collect::<String>().as_str() {
            "month" | "mo" => "month",
            "year" | "yr" => "year",
            "week" | "wk" => "week",
            "day" => "day",
            "hour" | "hr" => "hour",
            _ => return None,
        };
        if word(t.at(k)) {
            return None;
        }
        let unit = if n.is_one() { currency.to_string() } else { format!("{currency}s") };
        return Some((k - i, format!("{} {} a {}", spoken(&n), unit, period)));
    }
    None
}

/// "m²" is handled with its unit; "x²" (x squared) and "x³" (x cubed) here.
fn superscript_rule(t: &Text, i: usize) -> Option<(usize, String)> {
    let ch = t.at(i);
    if !ch.is_alphabetic() || word(t.at(i + 2)) {
        return None;
    }
    match t.at(i + 1) {
        '²' => Some((2, format!("{ch} squared"))),
        '³' => Some((2, format!("{ch} cubed"))),
        _ => None,
    }
}

/// Roman numerals after a cue word: "Chapter IV" (four), "World War II" (two), "Henry VIII" (the Eighth).
fn roman_rule(t: &Text, i: usize, out: &str) -> Option<(usize, String)> {
    let mut k = i;
    while t.at(k).is_ascii_uppercase() {
        k += 1;
    }
    if word(t.at(k)) {
        return None;
    }
    let s: String = (i..k).map(|p| t.at(p)).collect();
    let value = roman_value(&s)?;
    let prev = tail_words(out).first().copied().unwrap_or("");
    let cue = CARDINAL_CUES.iter().any(|c| c.eq_ignore_ascii_case(prev));
    if (s.len() >= 2 && cue) || (s == "I" && CAPITAL_CUES.contains(&prev)) {
        return Some((k - i, cardinal(value)));
    }
    if MONARCHS.contains(&prev) && s.len() >= 2 {
        return Some((k - i, format!("the {}", capitalised(&ordinal(value)))));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::rewrite;

    fn r(s: &str) -> String {
        rewrite(s)
    }

    #[test]
    fn units_take_singular_for_one_and_plural_otherwise() {
        assert_eq!(r("The bridge is 5 km long."), "The bridge is 5 kilometers long.");
        assert_eq!(r("Use 1 kWh per day."), "Use 1 kilowatt hour per day.");
        assert_eq!(r("A 2 TB drive is enough."), "A 2 terabyte drive is enough.");
        assert_eq!(r("The plot is 5 m² in area."), "The plot is 5 square meters in area.");
        assert_eq!(r("It is 3 mi ahead."), "It is 3 miles ahead.");
        assert_eq!(r("Copy at 100 MB/s."), "Copy at 100 megabytes per second.");
        assert_eq!(r("Speed limit: 65 mph."), "Speed limit: sixty-five miles per hour.");
    }

    #[test]
    fn units_leave_other_words_and_ambiguous_forms_alone() {
        assert_eq!(r("I can 5 mile walk."), "I can 5 mile walk.");
        assert_eq!(r("Cut it to 12 in. long."), "Cut it to 12 in. long.");
        assert_eq!(r("It is 5 a.m."), "It is 5 A M.");
        assert_eq!(r("The room has 5 chairs and 7 tables."), "The room has 5 chairs and 7 tables.");
    }

    #[test]
    fn ranges_and_ratios_read_as_to() {
        assert_eq!(r("We need 5-10 volunteers."), "We need 5 to 10 volunteers.");
        assert_eq!(r("Expect 10–20 guests."), "Expect 10 to 20 guests.");
        assert_eq!(r("The score was 3-1."), "The score was 3 to 1.");
        assert_eq!(r("The ratio is 16:9."), "The ratio is 16 to 9.");
        assert_eq!(r("The war lasted 1914-1918."), "The war lasted 1914 to 1918.");
        assert_eq!(r("We need 5 - 10 volunteers."), "We need 5 to 10 volunteers.");
        assert_eq!(r("Ages 6 – 12 are welcome."), "Ages 6 to 12 are welcome.");
    }

    #[test]
    fn ranges_leave_dates_phones_words_and_sums_alone() {
        assert_eq!(r("COVID-19 and B2B sales."), "COVID-19 and B2B sales.");
        assert_eq!(r("A 24-hour news piece and a 3-year-old dog."), "A 24-hour news piece and a 3-year-old dog.");
        assert_eq!(r("Subtract ten: 10 - 3 = 7."), "Subtract ten: 10 - 3 = 7.");
        assert_eq!(r("Compute 10 - 3 = 7 and 5 - 2 = 3."), "Compute 10 - 3 = 7 and 5 - 2 = 3.");
        assert_eq!(r("Call 555-0134 before noon."), "Call five five five zero one three four before noon.");
    }

    #[test]
    fn digit_strings_are_read_digit_by_digit() {
        assert_eq!(r("Code 0042 was entered."), "Code zero zero four two was entered.");
        assert_eq!(r("Her ZIP is 02139."), "Her ZIP is zero two one three nine.");
        assert_eq!(r("Your PIN is 4821."), "Your PIN is four eight two one.");
        assert_eq!(r("Call 911 now."), "Call nine one one now.");
        assert_eq!(r("Call 800-555-0199 for a quote."), "Call eight zero zero five five five zero one nine nine for a quote.");
        assert_eq!(r("Room 207 is booked."), "Room 207 is booked.");
    }

    #[test]
    fn dates_are_read_as_written_dates() {
        assert_eq!(r("We meet on 2026-11-05 at the lab."), "We meet on November 5th twenty twenty-six at the lab.");
        assert_eq!(r("Pay by 03/14/2027."), "Pay by March 14th twenty twenty-seven.");
        assert_eq!(r("The deadline is 11/05/2026."), "The deadline is 11/05/2026.");
        assert_eq!(r("The date is 5 Nov 2026."), "The date is the 5th of November twenty twenty-six.");
        assert_eq!(r("Ship it on Dec. 25 at the latest."), "Ship it on Dec. twenty-fifth at the latest.");
        assert_eq!(r("The party was on December 25, 2026."), "The party was on December 25, 2026.");
        assert_eq!(r("The show opens on September 3rd."), "The show opens on September 3rd.");
    }

    #[test]
    fn money_with_a_scale_is_read_in_words() {
        assert_eq!(r("The company raised $2.5M in May."), "The company raised two point five million dollars in May.");
        assert_eq!(r("Rent is $300k a year."), "Rent is 300 thousand dollars a year.");
        assert_eq!(r("A pen costs 50¢."), "A pen costs 50 cents.");
        assert_eq!(r("The fee is $5/month."), "The fee is 5 dollars a month.");
        assert_eq!(r("Lunch cost $5.50."), "Lunch cost $5.50.");
    }

    #[test]
    fn degrees_and_clock_times() {
        assert_eq!(r("It is 75°F today."), "It is seventy-five degrees Fahrenheit today.");
        assert_eq!(r("The city sits at 20° north."), "The city sits at 20 degrees north.");
        assert_eq!(r("Make a 90° turn."), "Make a 90 degree turn.");
        assert_eq!(r("The show starts at 8:05."), "The show starts at 8 oh five.");
        assert_eq!(r("Please arrive by 9:00 sharp."), "Please arrive by 9 o'clock sharp.");
        assert_eq!(r("Lunch is at 11:30 a.m."), "Lunch is at 11 thirty A M.");
        assert_eq!(r("The train leaves at 12:30 today."), "The train leaves at 12:30 today.");
    }

    #[test]
    fn fractions_roman_numerals_and_multiplication() {
        assert_eq!(r("Add 1/2 cup of milk."), "Add one half cup of milk.");
        assert_eq!(r("Take 2/3 of the total."), "Take two thirds of the total.");
        assert_eq!(r("Pour ½ cup of oil."), "Pour one half cup of oil.");
        assert_eq!(r("The 1/2/2026 form."), "The 1/2/2026 form.");
        assert_eq!(r("The 9/11 attacks changed things."), "The 9/11 attacks changed things.");
        assert_eq!(r("It costs €18.75 today."), "It costs €18.75 today.");
        assert_eq!(r("It costs €20 today."), "It costs 20 euros today.");
        assert_eq!(r("Henry VIII had six wives."), "Henry the Eighth had six wives.");
        assert_eq!(r("She read Chapter IV twice."), "She read Chapter four twice.");
        assert_eq!(r("Super Bowl LV was big."), "Super Bowl fifty-five was big.");
        assert_eq!(r("The book is in volume III."), "The book is in volume three.");
        assert_eq!(r("I read the book I bought."), "I read the book I bought.");
        assert_eq!(r("The 16th of July, 1969 was hot."), "The 16th of July, nineteen sixty-nine was hot.");
        assert_eq!(r("It is the 21st of the month."), "It is the 21st of the month.");
        assert_eq!(r("I went home early."), "I went home early.");
        assert_eq!(r("Compute 3 × 4 = 12."), "Compute 3 times 4 = 12.");
        assert_eq!(r("The room is 4 × 5 meters."), "The room is 4 by 5 meters.");
        assert_eq!(r("The new chip is 2x faster."), "The new chip is 2 times faster.");
        assert_eq!(r("The new chip is 2 × faster."), "The new chip is 2 times faster.");
        assert_eq!(r("Solve for 2x."), "Solve for 2x.");
        assert_eq!(r("Let x² be the variable."), "Let x squared be the variable.");
    }

    #[test]
    fn ratio_cue_must_be_a_whole_word() {
        assert_eq!(r("The train leaves at 12:30 with spare parts."), "The train leaves at 12:30 with spare parts.");
        assert_eq!(r("Mix 1:20 with water."), "Mix 1 to 20 with water.");
        assert_eq!(r("The summer sale starts at 10 a.m."), "The summer sale starts at 10 A M.");
    }

    #[test]
    fn arbitrary_input_never_panics() {
        // A fixed pseudo-random sequence over the characters the rules look at.
        const ALPHABET: &[char] = &[
            '0', '1', '2', '3', '5', '9', '1', '4', ' ', ' ', ',', '.', ':', '/', '-', '–', '$', '£', '€', '¢', '%', '°', '²', '³',
            '½', '¼', '¾', '×', 'x', 'a', 'A', 'm', 'M', 'p', 'P', 'I', 'V', 'X', 'k', 'B', 'n', 'o', 'f', 'C', 'F', 'K', '(', ')',
            '+', '=', '\'', 'é', '\n', 'ß', '中', 'D', 'e', 's', 't', 'h', 'r',
        ];
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..4000 {
            let mut s = String::new();
            for _ in 0..(seed % 24) {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                s.push(ALPHABET[((seed >> 33) as usize) % ALPHABET.len()]);
            }
            let once = r(&s);
            assert_eq!(r(&once).is_empty(), once.is_empty());
        }
    }

    #[test]
    fn plain_sentences_and_edge_inputs_do_not_change() {
        for s in ["", " ", "No, thanks.", "Dr. Smith will see you at noon.", "I live in the U.S.", "2026", "ü 5 ß"] {
            assert_eq!(r(s), s);
        }
        let long = "5 ".repeat(5000);
        assert!(!r(&long).is_empty());
        assert_eq!(r("9"), "9");
        assert_eq!(r("5-"), "5-");
        assert_eq!(r("3 ×"), "3 ×");
        assert_eq!(r("12:"), "12:");
    }
}
