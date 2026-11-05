//! English number words in the reading conventions misaki uses: cardinal, ordinal, year and decimal forms.
//! Written from the English number rules.

/// A natural number as decimal digits without leading zeros.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Big(String);

impl Big {
    /// Drops leading zeros; a string of zeros alone becomes "0". Any other character is kept, and the number
    /// functions answer None for it.
    pub fn parse(s: &str) -> Big {
        let rest = s.trim_start_matches('0');
        Big(if rest.is_empty() && !s.is_empty() { "0".to_string() } else { rest.to_string() })
    }

    fn digits(&self) -> Option<&str> {
        let s = self.0.as_str();
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then_some(s)
    }
}

/// The largest digit count that has a name: 102 groups of three digits, the top group being centillion (10^303).
const MAX_DIGITS: usize = 306;

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];

const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

// Scale names for 10^(3i), i = 0 to 101: the standard Latin short-scale names (Conway-Guy formation).
const SCALE: [&str; 102] = [
    "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion",
    "septillion", "octillion", "nonillion", "decillion", "undecillion", "duodecillion", "tredecillion",
    "quattuordecillion", "quindecillion", "sexdecillion", "septdecillion", "octodecillion",
    "novemdecillion", "vigintillion", "unvigintillion", "duovigintillion", "tresvigintillion",
    "quattuorvigintillion", "quinvigintillion", "sesvigintillion", "septemvigintillion",
    "octovigintillion", "novemvigintillion", "trigintillion", "untrigintillion", "duotrigintillion",
    "trestrigintillion", "quattuortrigintillion", "quintrigintillion", "sestrigintillion",
    "septentrigintillion", "octotrigintillion", "noventrigintillion", "quadragintillion",
    "unquadragintillion", "duoquadragintillion", "tresquadragintillion", "quattuorquadragintillion",
    "quinquadragintillion", "sesquadragintillion", "septenquadragintillion", "octoquadragintillion",
    "novenquadragintillion", "quinquagintillion", "unquinquagintillion", "duoquinquagintillion",
    "tresquinquagintillion", "quattuorquinquagintillion", "quinquinquagintillion",
    "sesquinquagintillion", "septenquinquagintillion", "octoquinquagintillion",
    "novenquinquagintillion", "sexagintillion", "unsexagintillion", "duosexagintillion",
    "tresexagintillion", "quattuorsexagintillion", "quinsexagintillion", "sesexagintillion",
    "septensexagintillion", "octosexagintillion", "novemsexagintillion", "septuagintillion",
    "unseptuagintillion", "duoseptuagintillion", "treseptuagintillion", "quattuorseptuagintillion",
    "quinseptuagintillion", "seseptuagintillion", "septenseptuagintillion", "octoseptuagintillion",
    "novemseptuagintillion", "octogintillion", "unoctogintillion", "duooctogintillion",
    "tresoctogintillion", "quattuoroctogintillion", "quinoctogintillion", "sexoctogintillion",
    "septenoctogintillion", "octooctogintillion", "novemoctogintillion", "nonagintillion",
    "unnonagintillion", "duononagintillion", "trenonagintillion", "quattuornonagintillion",
    "quinnonagintillion", "sexnonagintillion", "septennonagintillion", "octononagintillion",
    "novemnonagintillion", "centillion",
];

/// Cardinal words for a natural number of at most 306 digits; None above that.
pub fn cardinal(value: &Big) -> Option<String> {
    let digits = value.digits()?;
    if digits.len() > MAX_DIGITS {
        return None;
    }
    // Groups of three digits, lowest first, so that the index of a group is its scale index.
    let groups: Vec<u32> = digits.as_bytes().rchunks(3).map(group_value).collect();
    let mut words = String::new();
    for (i, &g) in groups.iter().enumerate().rev() {
        if g == 0 {
            continue;
        }
        if !words.is_empty() {
            // "and" comes only before a final group under one hundred: "one thousand and five", but
            // "one thousand, five hundred".
            words.push_str(if i == 0 && g < 100 { " and " } else { ", " });
        }
        words.push_str(&below_1000(g));
        if i > 0 {
            words.push(' ');
            words.push_str(SCALE[i]);
        }
    }
    Some(if words.is_empty() { "zero".to_string() } else { words })
}

/// Ordinal words: the cardinal words with the last word in its ordinal form ("one hundred and first").
pub fn ordinal(value: &Big) -> Option<String> {
    let words = cardinal(value)?;
    let cut = words.rfind(|c: char| !c.is_ascii_lowercase()).map_or(0, |i| i + 1);
    Some(format!("{}{}", &words[..cut], ordinal_word(&words[cut..])))
}

fn ordinal_word(word: &str) -> String {
    match word {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        _ => match word.strip_suffix('y') {
            Some(stem) => format!("{stem}ieth"), // twenty, twentieth
            None => format!("{word}th"),
        },
    }
}

/// Year words. Three- and four-digit numbers are said as a high part and a low part ("nineteen hundred",
/// "nineteen oh-five", "twenty ten"), except that X000 to X009 read as cardinals ("two thousand and five").
/// Numbers of other lengths read as cardinals.
pub fn year(value: &Big) -> Option<String> {
    let digits = value.digits()?;
    if !(3..=4).contains(&digits.len()) {
        return cardinal(value);
    }
    let n: u32 = digits.parse().ok()?;
    if n >= 1000 && n % 1000 < 10 {
        return cardinal(value);
    }
    let (high, low) = (below_100(n / 100), n % 100);
    Some(match low {
        0 => format!("{high} hundred"),
        1..=9 => format!("{high} oh-{}", below_100(low)),
        _ => format!("{high} {}", below_100(low)),
    })
}

/// Words for a decimal string read as a Python float: the whole part as a cardinal, then "point" and each digit after
/// the point. Python's repr decides how many digits follow the point (1e-5 has five, 85.0 none). The fraction is
/// taken in floating point: a result within 0.01 of a whole number counts as that number, and anything else is
/// truncated. Zeros inside the fraction are read, so 0.00001 reads "zero point zero zero zero zero one".
pub fn float_words(text: &str) -> Option<String> {
    let x: f64 = text.parse().ok()?;
    if !x.is_finite() {
        return None;
    }
    let a = x.abs();
    // The shortest round-trip digits in scientific form, such as "1.90460958967481e8": the digits after the point
    // are the digit count, less one, less the exponent.
    let sci = format!("{a:e}");
    let (mantissa, exp) = sci.split_once('e')?;
    let exp: i64 = exp.parse().ok()?;
    let digit_count = mantissa.bytes().filter(u8::is_ascii_digit).count() as i64;
    let places = (digit_count - 1 - exp).max(0) as usize;
    let scale: f64 = format!("1e{places}").parse().ok()?;
    if !scale.is_finite() {
        return None;
    }
    let whole = a.trunc();
    let mut part = (a - whole) * scale;
    if (part - part.round()).abs() < 0.01 {
        part = part.round();
    }
    let digits = format!("{:.0}", part.trunc());
    let mut words = cardinal(&Big::parse(&format!("{whole:.0}")))?;
    if digits != "0" {
        words.push_str(" point");
        for b in format!("{digits:0>places$}").bytes() {
            words.push(' ');
            words.push_str(ONES[usize::from(b - b'0')]);
        }
    }
    Some(if x < 0.0 { format!("minus {words}") } else { words })
}

/// Words for 0 to 99, with a hyphen between the tens and the units.
fn below_100(n: u32) -> String {
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    let tens = TENS[(n / 10) as usize];
    match n % 10 {
        0 => tens.to_string(),
        units => format!("{tens}-{}", ONES[units as usize]),
    }
}

/// Words for 0 to 999; "and" comes between the hundreds and a tens-and-units part.
fn below_1000(n: u32) -> String {
    match (n / 100, n % 100) {
        (0, rest) => below_100(rest),
        (hundreds, 0) => format!("{} hundred", ONES[hundreds as usize]),
        (hundreds, rest) => format!("{} hundred and {}", ONES[hundreds as usize], below_100(rest)),
    }
}

/// The value of up to three decimal digits.
fn group_value(digits: &[u8]) -> u32 {
    digits.iter().fold(0, |v, &b| v * 10 + u32::from(b - b'0'))
}
