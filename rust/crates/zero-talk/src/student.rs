//! The student (384 wide, FFN-384 form): a phoneme text encoder with a duration head, a frame block that turns expanded
//! features into F0 and energy, and the acoustic block that produces the generator's 512-channel latents at 80 Hz. Activations are
//! channel-major [c][t], like the generator's.
//!
//! The text encoder reads a whole sentence (its blocks look 3 frames ahead). The frame path is causal: the frame block and the
//! acoustic block keep a 6-column history, a running energy and (for the delta mixers) a recurrent memory, so they stream.
use crate::archive::{Part, Result};
use crate::kernels::{gemm, Act, Backend};
use crate::plan::{fold_combine, COORDS};

const LN_EPS: f32 = 1e-5;

/// y = W x + b for x [in][t].
pub struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    n_out: usize,
    n_in: usize,
}

impl Linear {
    fn load(ar: &Part, p: &str) -> Result<Self> {
        let shape = ar.shape(&format!("{p}.weight"))?;
        Ok(Linear { w: ar.f32(&format!("{p}.weight"))?, b: ar.f32(&format!("{p}.bias"))?, n_out: shape[0], n_in: shape[1] })
    }

    pub fn apply(&self, be: Backend, x: &Act) -> Act {
        assert_eq!(x.c, self.n_in);
        let mut y = Act::zeros(self.n_out, x.t);
        for o in 0..self.n_out {
            y.row_mut(o).fill(self.b[o]);
        }
        if x.t > 0 {
            unsafe { gemm(be, self.n_out, x.t, self.n_in, self.w.as_ptr(), self.n_in, x.data.as_ptr(), x.t, 1.0,
                          y.data.as_mut_ptr(), x.t) };
        }
        y
    }
}

/// LayerNorm over the channels of every column.
struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn load(ar: &Part, p: &str) -> Result<Self> {
        Ok(LayerNorm { w: ar.f32(&format!("{p}.weight"))?, b: ar.f32(&format!("{p}.bias"))? })
    }

    fn apply(&self, x: &Act) -> Act {
        let (c, t) = (x.c, x.t);
        let (mut mean, mut var) = (vec![0.0f64; t], vec![0.0f64; t]);
        for i in 0..c {
            for (m, v) in mean.iter_mut().zip(x.row(i)) {
                *m += *v as f64;
            }
        }
        mean.iter_mut().for_each(|m| *m /= c as f64);
        for i in 0..c {
            for ((s, m), v) in var.iter_mut().zip(&mean).zip(x.row(i)) {
                *s += (*v as f64 - m).powi(2);
            }
        }
        let rstd: Vec<f64> = var.iter().map(|s| 1.0 / (s / c as f64 + LN_EPS as f64).sqrt()).collect();
        let mut y = Act::zeros(c, t);
        for i in 0..c {
            let (w, b) = (self.w[i] as f64, self.b[i] as f64);
            for (j, (o, v)) in y.row_mut(i).iter_mut().zip(x.row(i)).enumerate() {
                *o = ((*v as f64 - mean[j]) * rstd[j] * w + b) as f32;
            }
        }
        y
    }
}

/// Exact GELU, x * Phi(x), with Phi from the rational Chebyshev erfc of Numerical Recipes (fractional error < 1.2e-7).
#[inline(always)]
fn gelu(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.5 * a);
    let poly = -a * a - 1.265_512_2
        + t * (1.000_023_7 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07
        + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc_a = t * poly.exp();
    // Phi(x) = erfc(-z) / 2: for z >= 0 it is 1 - erfc(z) / 2, otherwise erfc(|z|) / 2.
    x * if z >= 0.0 { 1.0 - 0.5 * erfc_a } else { 0.5 * erfc_a }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One causal ConvNeXt block: depthwise conv (kernel 7), LayerNorm, up projection, GELU, a cumulative-energy response
/// and the down projection. `energy` and the column history are the block's streaming state.
struct Block {
    dw_w: Vec<f32>, // [c][kernel]
    dw_b: Vec<f32>,
    norm: LayerNorm,
    up: Linear,
    down: Linear,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    kernel: usize,
}

#[derive(Clone)]
struct BlockState {
    hist: Act,       // the last kernel - 1 input columns
    energy: Vec<f32>, // the running sum of squared up-projection activations, per hidden channel
}

impl Block {
    fn load(ar: &Part, p: &str, kernel: usize) -> Result<Self> {
        Ok(Block {
            dw_w: ar.f32(&format!("{p}.depthwise.weight"))?,
            dw_b: ar.f32(&format!("{p}.depthwise.bias"))?,
            norm: LayerNorm::load(ar, &format!("{p}.norm"))?,
            up: Linear::load(ar, &format!("{p}.up"))?,
            down: Linear::load(ar, &format!("{p}.down"))?,
            gamma: ar.f32(&format!("{p}.gamma"))?,
            beta: ar.f32(&format!("{p}.beta"))?,
            kernel,
        })
    }

    fn fresh(&self) -> BlockState {
        BlockState { hist: Act::zeros(self.dw_b.len(), self.kernel - 1), energy: vec![0.0; self.gamma.len()] }
    }

    /// `joined` holds the depthwise input: kernel - 1 columns of context, then x's columns (t + kernel - 1 in all).
    fn forward(&self, be: Backend, x: &Act, joined: &Act, energy_in: &[f32]) -> (Act, Vec<f32>) {
        let (c, t, k) = (x.c, x.t, self.kernel);
        let mut h = Act::zeros(c, t);
        for ch in 0..c {
            let (w, src) = (&self.dw_w[ch * k..(ch + 1) * k], joined.row(ch));
            for (i, o) in h.row_mut(ch).iter_mut().enumerate() {
                let mut acc = self.dw_b[ch];
                for j in 0..k {
                    acc += w[j] * src[i + j];
                }
                *o = acc;
            }
        }
        let mut h = self.up.apply(be, &self.norm.apply(&h));
        h.map(gelu);
        // response = sqrt(cumulative energy + 1e-6), normalized by its mean over channels (per column).
        let (hc, mut cum) = (h.c, Act::zeros(h.c, t));
        for ch in 0..hc {
            let mut run = 0.0f32;
            for (o, v) in cum.row_mut(ch).iter_mut().zip(h.row(ch)) {
                run += v * v;
                *o = run + energy_in[ch];
            }
        }
        let mut scale = vec![0.0f32; t];
        for ch in 0..hc {
            for (s, v) in scale.iter_mut().zip(cum.row(ch)) {
                *s += (v + 1e-6).sqrt();
            }
        }
        scale.iter_mut().for_each(|s| *s = *s / hc as f32 + 1e-6);
        for ch in 0..hc {
            let (g, b) = (self.gamma[ch], self.beta[ch]);
            let cr = cum.row(ch);
            for (j, v) in h.row_mut(ch).iter_mut().enumerate() {
                let response = (cr[j] + 1e-6).sqrt() / scale[j];
                *v = *v + g * *v * response + b;
            }
        }
        let mut y = self.down.apply(be, &h);
        y.add(x);
        let energy = (0..hc).map(|ch| cum.row(ch)[t - 1]).collect();
        (y, energy)
    }

    /// Causal step: the block's state supplies the left context.
    fn step(&self, be: Backend, st: &mut BlockState, x: &Act) -> Act {
        let joined = std::mem::take(&mut st.hist).cat(x);
        let (y, energy) = self.forward(be, x, &joined, &st.energy);
        st.hist = joined.columns(joined.t - (self.kernel - 1), joined.t);
        st.energy = energy;
        y
    }
}

/// Gated delta-rule mixer (6 heads of 64): a linear-attention memory per head with learned decay and write gates.
struct Mixer {
    norm: LayerNorm,
    qkv: Linear,
    decay_write: Linear,
    gate: Linear,
    out_norm: LayerNorm,
    out: Linear,
    heads: usize,
    dim: usize,
}

impl Mixer {
    fn load(ar: &Part, p: &str, heads: usize, dim: usize) -> Result<Self> {
        Ok(Mixer {
            norm: LayerNorm::load(ar, &format!("{p}.norm"))?,
            qkv: Linear::load(ar, &format!("{p}.qkv"))?,
            decay_write: Linear::load(ar, &format!("{p}.decay_write"))?,
            gate: Linear::load(ar, &format!("{p}.gate"))?,
            out_norm: LayerNorm::load(ar, &format!("{p}.out_norm"))?,
            out: Linear::load(ar, &format!("{p}.out"))?,
            heads,
            dim,
        })
    }

    /// `memory`: [head][key][value], updated in place.
    fn step(&self, be: Backend, memory: &mut [f32], x: &Act) -> Act {
        let (h, d, t) = (self.heads, self.dim, x.t);
        let hn = self.norm.apply(x);
        let qkv = self.qkv.apply(be, &hn);
        let gates = self.decay_write.apply(be, &hn);
        let mut y = Act::zeros(h * d, t);
        let (mut q, mut k, mut v, mut res, mut acc) = (vec![0f32; d], vec![0f32; d], vec![0f32; d], vec![0f32; d], vec![0f32; d]);
        for step in 0..t {
            for head in 0..h {
                let col = |which: usize, i: usize| qkv.row(which * h * d + head * d + i)[step];
                let (mut qn, mut kn) = (0f32, 0f32);
                for i in 0..d {
                    q[i] = col(0, i);
                    k[i] = col(1, i);
                    v[i] = col(2, i);
                    qn += q[i] * q[i];
                    kn += k[i] * k[i];
                }
                let (qn, kn) = (1.0 / qn.sqrt().max(1e-12), 1.0 / kn.sqrt().max(1e-12));
                q.iter_mut().for_each(|a| *a *= qn);
                k.iter_mut().for_each(|a| *a *= kn);
                let (alpha, beta) = (sigmoid(gates.row(head)[step]), sigmoid(gates.row(h + head)[step]));
                let m = &mut memory[head * d * d..(head + 1) * d * d];
                // M = alpha M;  residual = v - k^T M;  M += beta k residual^T;  y = q^T M
                acc.fill(0.0);
                for (i, row) in m.chunks_exact_mut(d).enumerate() {
                    for (a, r) in acc.iter_mut().zip(row.iter_mut()) {
                        *r *= alpha;
                        *a += k[i] * *r;
                    }
                }
                for i in 0..d {
                    res[i] = v[i] - acc[i];
                }
                acc.fill(0.0);
                for (i, row) in m.chunks_exact_mut(d).enumerate() {
                    let wk = beta * k[i];
                    for ((a, r), e) in acc.iter_mut().zip(row.iter_mut()).zip(&res) {
                        *r += wk * *e;
                        *a += q[i] * *r;
                    }
                }
                for i in 0..d {
                    y.data[(head * d + i) * t + step] = acc[i];
                }
            }
        }
        // Per-head LayerNorm over the 64 values, then the SiLU gate.
        let mut yn = Act::zeros(h * d, t);
        for head in 0..h {
            let part = y.rows(head * d, (head + 1) * d);
            let normed = self.out_norm.apply(&part);
            yn.data[head * d * t..(head + 1) * d * t].copy_from_slice(&normed.data);
        }
        let mut g = self.gate.apply(be, &hn);
        g.map(|a| a * sigmoid(a));
        for (a, b) in yn.data.iter_mut().zip(&g.data) {
            *a *= *b;
        }
        let mut out = self.out.apply(be, &yn);
        out.add(x);
        out
    }
}

pub struct TextToAcoustic {
    width: usize,
    lookahead: usize,
    embed: Vec<f32>, // [vocab][width]
    voice_fold: Vec<f32>, // [1 + COORDS][width]: the bias every token embedding gets, at the mean row and per length coordinate
    text: Vec<Block>,
    duration: Linear,
    phase: Linear,
    frame_block: Block,
    prosody: Linear,
    condition: Linear,
    acoustic: Vec<Block>,
    mixers: Vec<Option<Mixer>>, // after block i
    output: Linear,
}

/// The acoustic blocks' streaming state: every block's history and energy, and the two mixers' memories.
pub struct AcousticState {
    blocks: Vec<BlockState>,
    memories: Vec<Vec<f32>>,
}

impl AcousticState {
}

impl TextToAcoustic {
    pub fn load(ar: &Part) -> Result<Self> {
        let get = |k: &str| ar.meta_as::<usize>(&format!("student_{k}"));
        let (width, kernel, heads, head_dim) = (get("width")?, get("kernel")?, get("heads")?, get("head_dim")?);
        let (text_depth, acoustic_depth, lookahead) = (get("text_depth")?, get("acoustic_depth")?, get("text_lookahead")?);
        if get("expansion")? != 1 || acoustic_depth != 6 || heads * head_dim != width || lookahead % text_depth != 0
            || lookahead / text_depth > kernel - 1 {
            return Err("unsupported student configuration".into());
        }
        let blocks = |name: &str, n: usize| (0..n).map(|i| Block::load(ar, &format!("{name}.{i}"), kernel)).collect::<Result<Vec<_>>>();
        let mixers = (0..acoustic_depth)
            .map(|i| {
                let p = format!("acoustic.mixers.{i}");
                ar.has(&format!("{p}.qkv.weight")).then(|| Mixer::load(ar, &p, heads, head_dim)).transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TextToAcoustic {
            width,
            lookahead: lookahead / text_depth,
            embed: ar.f32("prosody.embedding.weight")?,
            voice_fold: ar.f32("prosody.voice_fold")?,
            text: blocks("prosody.blocks", text_depth)?,
            duration: Linear::load(ar, "prosody.duration")?,
            phase: Linear::load(ar, "prosody.phase")?,
            frame_block: Block::load(ar, "prosody.frame_block", kernel)?,
            prosody: Linear::load(ar, "prosody.prosody")?,
            condition: Linear::load(ar, "acoustic.condition")?,
            acoustic: blocks("acoustic.blocks", acoustic_depth)?,
            mixers,
            output: Linear::load(ar, "acoustic.output")?,
        })
    }

    /// Whole-sentence text encoder: ids, the sentence's length coordinates -> features [width][n], predicted durations (40 Hz
    /// units, before rounding).
    pub fn encode(&self, be: Backend, ids: &[usize], coords: &[f32; COORDS]) -> (Act, Vec<f32>) {
        let (w, n) = (self.width, ids.len());
        let s = fold_combine(&self.voice_fold, coords);
        let mut x = Act::zeros(w, n);
        for c in 0..w {
            for (t, &id) in ids.iter().enumerate() {
                x.data[c * n + t] = self.embed[id * w + c] + s[c] as f32;
            }
        }
        for b in &self.text {
            // Zero-padded context: kernel - 1 - lookahead columns before, `lookahead` after.
            let (before, after) = (b.kernel - 1 - self.lookahead, self.lookahead);
            let mut joined = Act::zeros(w, before + x.t + after);
            for c in 0..w {
                joined.row_mut(c)[before..before + x.t].copy_from_slice(x.row(c));
            }
            x = b.forward(be, &x, &joined, &vec![0.0; b.gamma.len()]).0;
        }
        let dur = self.duration.apply(be, &x);
        (x, dur.data.iter().map(|&d| softplus(d)).collect())
    }

    /// Frame features, raw scaled F0 and raw energy for the expanded sentence. `counts40` are the 40 Hz token counts;
    /// every token spans twice as many 80 Hz frames.
    pub fn frames(&self, be: Backend, feats: &Act, counts40: &[usize]) -> (Act, Vec<f32>, Vec<f32>) {
        let (w, n) = (self.width, feats.t);
        let total: usize = counts40.iter().map(|c| 2 * c).sum();
        let pw = &self.phase.w;
        let mut x = Act::zeros(w, total);
        let mut f = 0;
        for tok in 0..n {
            let count = 2 * counts40[tok];
            for i in 0..count {
                let phase = i as f32 / count as f32;
                for c in 0..w {
                    x.data[c * total + f] = feats.data[c * n + tok] + (pw[c] * phase + self.phase.b[c]);
                }
                f += 1;
            }
        }
        let mut st = self.frame_block.fresh();
        let frames = self.frame_block.step(be, &mut st, &x);
        let p = self.prosody.apply(be, &frames);
        (frames, p.row(0).iter().map(|&v| softplus(v)).collect(), p.row(1).to_vec())
    }

    /// The streaming state of the acoustic blocks and mixers, before the first push.
    pub fn acoustic_state(&self) -> AcousticState {
        let d = self.mixers.iter().flatten().next().map_or(0, |m| m.dim);
        AcousticState {
            blocks: self.acoustic.iter().map(|b| b.fresh()).collect(),
            memories: self.mixers.iter().map(|m| m.as_ref().map_or(vec![], |m| vec![0.0; m.heads * d * d])).collect(),
        }
    }

    /// Latents [512][t] for the next frames: their features, (polished) scaled F0 and energy.
    pub fn push(&self, be: Backend, st: &mut AcousticState, frames: &Act, f0: &[f32], energy: &[f32]) -> Act {
        let t = frames.t;
        let mut cond = Act::zeros(2, t);
        cond.row_mut(0).copy_from_slice(f0);
        cond.row_mut(1).copy_from_slice(energy);
        let mut x = self.condition.apply(be, &cond);
        x.add(frames);
        for (i, b) in self.acoustic.iter().enumerate() {
            x = b.step(be, &mut st.blocks[i], &x);
            if let Some(m) = &self.mixers[i] {
                x = m.step(be, &mut st.memories[i], &x);
            }
        }
        self.output.apply(be, &x)
    }
}
