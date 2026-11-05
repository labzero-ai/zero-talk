//! The G2P network: a BiGRU encoder and a GRU decoder with dot-product attention, greedy decoding, one word at a time.
//! The arithmetic: float32 storage at the stored places, and float64 accumulation inside each dot product.
use std::cell::RefCell;
use std::collections::HashMap;

use crate::archive::{Archive, Result};

const BOS: usize = 1;
const EOS: usize = 2;

struct Gru {
    wih: Vec<f32>,
    whh: Vec<f32>,
    bih: Vec<f32>,
    bhh: Vec<f32>,
}

pub struct G2pNet {
    e: usize,
    h: usize,
    d: usize,
    v: usize,
    de: usize,
    enc_emb: Vec<f32>,
    enc_f: Gru,
    enc_b: Gru,
    init_w: Vec<f32>,
    init_b: Vec<f32>,
    dec_emb: Vec<f32>,
    dec: Gru,
    query: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    chars: Vec<char>,
    itos: Vec<String>,
    cache: RefCell<HashMap<String, Option<String>>>,
}

/// out[j] = bias[j] + sum_k w[j][k] * x[k], accumulated in float64, stored as float32.
fn mm(w: &[f32], rows: usize, cols: usize, x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    for j in 0..rows {
        let mut s = bias.map_or(0.0, |b| b[j] as f64);
        for (a, b) in w[j * cols..(j + 1) * cols].iter().zip(x) {
            s += *a as f64 * *b as f64;
        }
        out[j] = s as f32;
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One GRU step in place on h (gate order r, z, n); x has the layer's input width.
fn gru(g: &Gru, i: usize, hd: usize, x: &[f32], h: &mut [f32], gi: &mut [f32], gh: &mut [f32]) {
    mm(&g.wih, 3 * hd, i, x, Some(&g.bih), gi);
    mm(&g.whh, 3 * hd, hd, h, Some(&g.bhh), gh);
    for k in 0..hd {
        let r = sigmoid(gi[k] as f64 + gh[k] as f64);
        let z = sigmoid(gi[hd + k] as f64 + gh[hd + k] as f64);
        let n = (gi[2 * hd + k] as f64 + r * gh[2 * hd + k] as f64).tanh();
        h[k] = ((1.0 - z) * n + z * h[k] as f64) as f32;
    }
}

impl G2pNet {
    pub fn load(ar: &Archive) -> Result<Self> {
        let t = |name: &str| ar.f32(&format!("net.{name}"));
        let dim = |name: &str, axis: usize| ar.shape(&format!("net.{name}")).map(|s| s[axis]);
        let gru = |p: &str| -> Result<Gru> {
            Ok(Gru { wih: t(&format!("{p}.weight_ih_l0"))?, whh: t(&format!("{p}.weight_hh_l0"))?, bih: t(&format!("{p}.bias_ih_l0"))?, bhh: t(&format!("{p}.bias_hh_l0"))? })
        };
        let (e, h, d, v, de) = (dim("enc_emb.weight", 1)?, dim("enc_f.weight_hh_l0", 1)?, dim("dec.weight_hh_l0", 1)?, dim("out.weight", 0)?, dim("dec_emb.weight", 1)?);
        let chars: Vec<char> = ar.meta("net_chars")?.chars().collect();
        let mut itos: Vec<String> = ["<pad>", "<bos>", "<eos>"].iter().map(|s| s.to_string()).collect();
        itos.extend(ar.meta("net_phonemes")?.chars().map(String::from));
        let (n_in, n_out) = (dim("enc_emb.weight", 0)?, v);
        if n_in != chars.len() + 1 || n_out != itos.len() || dim("enc_f.weight_ih_l0", 0)? != 3 * h || dim("dec.weight_ih_l0", 0)? != 3 * d || dim("init_h.weight", 1)? != 2 * h {
            return Err("the network's weights do not fit its alphabets".into());
        }
        Ok(G2pNet { e, h, d, v, de, enc_emb: t("enc_emb.weight")?, enc_f: gru("enc_f")?, enc_b: gru("enc_b")?, init_w: t("init_h.weight")?, init_b: t("init_h.bias")?,
                    dec_emb: t("dec_emb.weight")?, dec: gru("dec")?, query: t("query.weight")?, out_w: t("out.weight")?, out_b: t("out.bias")?, chars, itos, cache: RefCell::new(HashMap::new()) })
    }

    /// The phonemes of a lowercase word (letters and apostrophes), remembered by word; None if the word has a character the
    /// network does not know.
    pub fn phonemes(&self, word: &str) -> Option<String> {
        if let Some(v) = self.cache.borrow().get(word) {
            return v.clone();
        }
        let v = self.decode(word).ok().map(|x| x.0);
        self.cache.borrow_mut().insert(word.to_string(), v.clone());
        v
    }

    /// Greedy decoding of one lowercase word (letters and apostrophes): the phonemes and the smallest gap between the best and
    /// second-best logit over the steps (a near-tie indicator).
    pub fn decode(&self, word: &str) -> Result<(String, f64)> {
        let (e, h, d, v, de) = (self.e, self.h, self.d, self.v, self.de);
        let h2 = 2 * h;
        let ids: Vec<usize> = word.chars().map(|c| self.chars.iter().position(|&x| x == c).map(|i| i + 1).ok_or_else(|| format!("character {c:?} of {word:?} is not in the network's alphabet"))).collect::<Result<_>>()?;
        let l = ids.len();
        if l == 0 {
            return Ok((String::new(), f64::INFINITY));
        }
        let (mut gi, mut gh) = (vec![0f32; 3 * h.max(d)], vec![0f32; 3 * h.max(d)]);
        let mut enc = vec![0f32; l * h2];
        for (dir, g) in [&self.enc_f, &self.enc_b].into_iter().enumerate() {
            let mut hs = vec![0f32; h];
            for s in 0..l {
                let t = if dir == 1 { l - 1 - s } else { s };
                gru(g, e, h, &self.enc_emb[ids[t] * e..(ids[t] + 1) * e], &mut hs, &mut gi, &mut gh);
                enc[t * h2 + dir * h..t * h2 + (dir + 1) * h].copy_from_slice(&hs);
            }
        }
        let mut pooled = vec![0f32; h2];
        for t in 0..l {
            for k in 0..h2 {
                pooled[k] = (pooled[k] as f64 + enc[t * h2 + k] as f64 / l as f64) as f32;
            }
        }
        let mut hid = vec![0f32; d];
        mm(&self.init_w, d, h2, &pooled, Some(&self.init_b), &mut hid);
        for x in hid.iter_mut() {
            *x = (*x as f64).tanh() as f32;
        }
        let (mut q, mut sc, mut z, mut lg) = (vec![0f32; h2], vec![0f32; l], vec![0f32; d + h2], vec![0f32; v]);
        let (mut tok, mut out, mut margin) = (BOS, String::new(), f64::INFINITY);
        for _ in 0..2 * l + 8 {
            gru(&self.dec, de, d, &self.dec_emb[tok * de..(tok + 1) * de], &mut hid, &mut gi, &mut gh);
            mm(&self.query, h2, d, &hid, None, &mut q);
            let mut mx = f32::NEG_INFINITY;
            for s in 0..l {
                let mut a = 0f64;
                for k in 0..h2 {
                    a += q[k] as f64 * enc[s * h2 + k] as f64;
                }
                sc[s] = a as f32;
                mx = mx.max(sc[s]);
            }
            let mut sum = 0f64;
            for x in sc.iter_mut() {
                *x = (*x as f64 - mx as f64).exp() as f32;
                sum += *x as f64;
            }
            z[..d].copy_from_slice(&hid);
            z[d..].fill(0.0);
            for s in 0..l {
                let a = sc[s] as f64 / sum;
                for k in 0..h2 {
                    z[d + k] = (z[d + k] as f64 + a * enc[s * h2 + k] as f64) as f32;
                }
            }
            mm(&self.out_w, v, d + h2, &z, Some(&self.out_b), &mut lg);
            let (mut best, mut v1, mut v2) = (0usize, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (k, &x) in lg.iter().enumerate() {
                let x = x as f64;
                if x > v1 {
                    v2 = v1;
                    v1 = x;
                    best = k;
                } else if x > v2 {
                    v2 = x;
                }
            }
            margin = margin.min(v1 - v2);
            if best == EOS {
                break;
            }
            out.push_str(&self.itos[best]);
            tok = best;
        }
        Ok((out, margin))
    }
}
