//! The part of the pipeline that is not a network: the phoneme vocabulary, the voice tables and the frame-level plan. The plan is
//! the neutral one: durations rounded with error diffusion, no tempo, pitch or energy rules, and F0 polished at consonant boundaries.
use std::collections::HashMap;

use crate::archive::{Part, Result};

const SPACE: usize = 16;
const SYNTHETIC: usize = 0;
const PUNCTUATION: [usize; 13] = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15];
// The unvoiced phonemes
const UNVOICED: [&str; 21] = ["h", "ʰ", "ʔ", "p", "t", "k", "ʈ", "c", "q", "f", "θ", "s", "ʃ", "ʂ", "ɕ", "ç", "χ", "ɸ", "ʧ",
                              "ʦ", "ʨ"];

pub struct Vocab {
    phonemes: Vec<String>,
    by_char: HashMap<char, usize>,
}

impl Vocab {
    pub fn load(ar: &Part) -> Result<Self> {
        let phonemes: Vec<String> = ar.meta("vocab")?.split('\x1f').map(str::to_string).collect();
        let by_char = phonemes.iter().enumerate().filter_map(|(i, p)| {
            let mut c = p.chars();
            c.next().filter(|_| c.next().is_none()).map(|c| (c, i))
        }).collect();
        Ok(Vocab { phonemes, by_char })
    }

    /// Number of ids (the largest id plus one).
    pub fn len(&self) -> usize {
        self.phonemes.len()
    }

    /// The tokenization: a pad id at each end and one id per phoneme character; characters outside the vocabulary are
    /// dropped and returned.
    pub fn encode(&self, phonemes: &str) -> (Vec<usize>, Vec<char>) {
        let (mut ids, mut dropped) = (vec![0], Vec::new());
        for c in phonemes.chars() {
            match self.by_char.get(&c) {
                Some(&i) => ids.push(i),
                None => dropped.push(c),
            }
        }
        ids.push(0);
        (ids, dropped)
    }

    pub fn phoneme(&self, id: usize) -> &str {
        self.phonemes.get(id).map_or("", |s| s.as_str())
    }

    pub fn is_punct(id: usize) -> bool {
        PUNCTUATION.contains(&id)
    }

    /// Unvoiced consonants, spaces and boundaries.
    pub fn is_unvoiced(&self, id: usize) -> bool {
        Self::is_punct(id) || id == SYNTHETIC || id == SPACE || UNVOICED.iter().any(|&u| u == self.phoneme(id))
    }
}

/// Length coordinates of the voice: each sentence length (phoneme count) has four coordinates, and everything that depends on
/// the voice (the generator's style layers, the student's embedding bias, the conditioner's style features) is folded into
/// constants plus one direction per coordinate. There is no style vector in the model.
pub const COORDS: usize = 4;

pub struct LengthCurve {
    coefficients: Vec<f32>, // [rows][COORDS]
}

impl LengthCurve {
    pub fn load(ar: &Part) -> Result<Self> {
        if ar.shape("coefficients")?[1] != COORDS {
            return Err(format!("voice.coefficients: expected {COORDS} coordinates per length"));
        }
        Ok(LengthCurve { coefficients: ar.f32("coefficients")? })
    }

    pub fn rows(&self) -> usize {
        self.coefficients.len() / COORDS
    }

    pub fn coords(&self, i: usize) -> [f32; COORDS] {
        self.coefficients[i * COORDS..(i + 1) * COORDS].try_into().unwrap()
    }
}

/// A folded affine term: `fold` is [1 + COORDS][n], the value at the mean row and one direction per coordinate; the result is
/// fold[0] + sum_r coords[r] * fold[1 + r], accumulated in float64.
pub fn fold_combine(fold: &[f32], coords: &[f32; COORDS]) -> Vec<f64> {
    let n = fold.len() / (1 + COORDS);
    (0..n).map(|i| fold[i] as f64 + (0..COORDS).map(|r| coords[r] as f64 * fold[(1 + r) * n + i] as f64).sum::<f64>()).collect()
}

/// Cumulative Bresenham duration rounding: durations (in 40 Hz units) become integers of at least 1, carrying the
/// rounding error forward (clamped to half a unit) and resetting it at punctuation.
pub fn bresenham(durations: &[f32], ids: &[usize]) -> Vec<usize> {
    let mut err = 0.0f64;
    durations.iter().zip(ids).map(|(&d, &id)| {
        if Vocab::is_punct(id) || id == SYNTHETIC {
            err = 0.0;
        }
        let target = d as f64 + err;
        let q = (target.round_ties_even() as i64).max(1);
        err = (target - q as f64).clamp(-0.5, 0.5);
        if Vocab::is_punct(id) {
            err = 0.0;
        }
        q as usize
    }).collect()
}

/// Consonant-onset F0 continuity clamp and boundary smoothing: inside
/// each run of unvoiced tokens that carries any F0, F0 is replaced by a line between its voiced neighbours (or held at
/// the one neighbour), then a Gaussian (sigma 1.75 frames) smooths the voiced frames.
pub fn onset_clamp(f0_in: &[f32], counts40: &[usize], ids: &[usize], vocab: &Vocab) -> Vec<f32> {
    const SIGMA: f64 = 1.75;
    let n = f0_in.len();
    if n < 4 {
        return f0_in.to_vec();
    }
    let f0: Vec<f64> = f0_in.iter().map(|&v| v as f64).collect();
    let starts: Vec<usize> = counts40.iter().scan(0, |a, c| { let s = *a; *a += 2 * c; Some(s) }).collect();
    let ends: Vec<usize> = starts.iter().zip(counts40).map(|(s, c)| s + 2 * c).collect();
    let unvoiced: Vec<bool> = ids.iter().map(|&i| vocab.is_unvoiced(i)).collect();
    if unvoiced.iter().all(|&u| u) {
        return f0_in.to_vec();
    }
    let mut f0c = f0.clone();
    let mean_valid = |s: &[f64]| -> Option<f64> {
        let v: Vec<f64> = s.iter().copied().filter(|&x| x > 30.0).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let nt = ids.len();
    let mut i = 0;
    while i < nt {
        if !unvoiced[i] {
            i += 1;
            continue;
        }
        let start_tok = i;
        while i < nt && unvoiced[i] {
            i += 1;
        }
        let end_tok = i;
        let (sf, ef) = (starts[start_tok], ends[end_tok - 1]);
        if f0[sf..ef].iter().all(|&v| v < 30.0) {
            continue;
        }
        let p_left = (start_tok > 0 && !unvoiced[start_tok - 1])
            .then(|| mean_valid(&f0[ends[start_tok - 1].saturating_sub(2)..ends[start_tok - 1]]))
            .flatten();
        let p_right = (end_tok < nt && !unvoiced[end_tok])
            .then(|| mean_valid(&f0[starts[end_tok]..n.min(starts[end_tok] + 2)]))
            .flatten();
        let len = ef - sf;
        match (p_left, p_right) {
            (Some(l), Some(r)) => {
                let step = 1.0 / (len + 1) as f64; // np.linspace(0, 1, len + 2)[1:-1]
                for k in 0..len {
                    let a = (k + 1) as f64 * step;
                    f0c[sf + k] = (1.0 - a) * l + a * r;
                }
            }
            (Some(v), None) | (None, Some(v)) => f0c[sf..ef].fill(v),
            (None, None) => {}
        }
    }
    let radius = (3.0 * SIGMA).ceil() as i64;
    let weights: Vec<f64> = (-radius..=radius).map(|k| (-0.5 * (k as f64 / SIGMA).powi(2)).exp()).collect();
    let active: Vec<bool> = f0c.iter().map(|&v| v > 30.0).collect();
    let mut out = f0c.clone();
    for t in (0..n).filter(|&t| active[t]) {
        let (mut wsum, mut acc) = (0.0f64, 0.0f64);
        for (kw, k) in (-radius..=radius).enumerate() {
            let idx = t as i64 + k;
            if idx >= 0 && (idx as usize) < n && active[idx as usize] {
                wsum += weights[kw];
                acc += weights[kw] * f0c[idx as usize];
            }
        }
        out[t] = acc / wsum;
    }
    out.iter().map(|&v| v as f32).collect()
}

/// The frame-level plan of one sentence, ready for the student.
pub struct Plan {
    pub f0_hz: Vec<f32>,
    pub scaled_f0: Vec<f32>,
    pub energy: Vec<f32>,
}

/// The neutral plan: the raw student F0 and energy, F0 polished, nothing else.
pub fn plan_neutral(counts40: Vec<usize>, raw_f0: &[f32], raw_energy: &[f32], ids: &[usize], vocab: &Vocab) -> Plan {
    let f0_np: Vec<f32> = raw_f0.iter().map(|&v| v * 200.0).collect();
    let f0_hz = onset_clamp(&f0_np, &counts40, ids, vocab);
    let scaled_f0 = f0_hz.iter().map(|&v| v / 200.0).collect();
    Plan { f0_hz, scaled_f0, energy: raw_energy.to_vec() }
}
