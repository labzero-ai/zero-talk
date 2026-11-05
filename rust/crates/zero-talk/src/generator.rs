//! The generator network after the harmonic source: an iSTFTNet-style topology with factorised convolutions, static norms and
//! bottleneck AdaIN. Latents x [512][F] at 80 Hz and the source spectrum har [22][60 F + 1] go in; the waveform [300 F] at 24 kHz
//! comes out.
//!
//! It runs as a stream with per-layer state: no window is recomputed, and the offline forward is one push.
//!
//! Based on kokoro 0.9.4's iSTFTNet (Apache-2.0), and changed for this engine.
use crate::archive::{Part, Result};
use crate::plan::{fold_combine, COORDS};
use crate::kernels::{affine_snake, pointwise, Act, Affine, Backend, Cols, Conv, ConvState, Istft};
use crate::source::COLUMNS_PER_FRAME;

/// A dense convolution or its Tucker factorization (1x1 to rank, K-tap core, 1x1 back with the bias).
enum Layer {
    Dense(Conv),
    Tucker(Conv, Conv, Conv),
}

impl Layer {
    /// `p`: the tensor-name prefix of the module; geometry of its (core) convolution.
    fn load(ar: &Part, p: &str, stride: usize, pad: usize, dil: usize, transposed: bool) -> Result<Self> {
        let conv = |name: &str, bias: Option<String>, stride, pad, dil, transposed| -> Result<Conv> {
            let bias = bias.map(|b| ar.f32(&b)).transpose()?;
            Ok(Conv::new(ar.f32(name)?, ar.shape(name)?, bias, stride, pad, dil, transposed))
        };
        if ar.has(&format!("{p}.weight")) {
            let bias = ar.has(&format!("{p}.bias")).then(|| format!("{p}.bias"));
            return Ok(Layer::Dense(conv(&format!("{p}.weight"), bias, stride, pad, dil, transposed)?));
        }
        Ok(Layer::Tucker(
            conv(&format!("{p}.first.weight"), None, 1, 0, 1, false)?,
            conv(&format!("{p}.core.weight"), None, stride, pad, dil, transposed)?,
            conv(&format!("{p}.last.weight"), Some(format!("{p}.last.bias")), 1, 0, 1, false)?,
        ))
    }

    /// The convolution that holds the layer's streaming state.
    fn core(&self) -> &Conv {
        match self {
            Layer::Dense(c) | Layer::Tucker(_, c, _) => c,
        }
    }

    fn cin(&self) -> usize {
        match self {
            Layer::Dense(c) | Layer::Tucker(c, _, _) => c.cin,
        }
    }

    fn run(&self, be: Backend, st: &mut ConvState, x: &Act, fin: bool) -> Act {
        match self {
            Layer::Dense(c) => st.run(be, c, x, fin),
            Layer::Tucker(first, core, last) => {
                let a = pointwise(be, first, x);
                pointwise(be, last, &st.run(be, core, &a, fin))
            }
        }
    }
}

/// One AdaIN site: static norm (moments set per utterance) and the folded style term.
struct Site {
    name: String,
    fold: Vec<f32>, // [1 + COORDS][2C]: gamma and beta at the mean row and per length coordinate
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
}

struct Unit {
    conv1: Layer,
    conv2: Layer,
    site1: usize,
    site2: usize,
    alpha: [Vec<f32>; 2],
    inv_alpha: [Vec<f32>; 2],
}

/// AdaINResBlock1: three units x = conv2(snake(adain2(conv1(snake(adain1(x)))))) + x.
struct ResBlock {
    units: Vec<Unit>,
}

/// A unit's stream state: both convolutions and the residual input waiting for their output.
struct UnitState {
    conv1: ConvState,
    conv2: ConvState,
    skip: Cols,
}

impl UnitState {
}

impl ResBlock {
    fn load(ar: &Part, p: &str, kernel: usize, dilations: &[usize], sites: &mut Vec<Site>) -> Result<Self> {
        let mut site = |name: String| -> Result<usize> {
            let fold = ar.f32(&format!("{name}.fold"))?;
            let norm_w = ar.f32(&format!("{name}.norm.weight"))?;
            if fold.len() != (1 + COORDS) * 2 * norm_w.len() {
                return Err(format!("{name}.fold: {} values, expected {} x 2 x {}", fold.len(), 1 + COORDS, norm_w.len()));
            }
            sites.push(Site {
                fold,
                norm_w,
                norm_b: ar.f32(&format!("{name}.norm.bias"))?,
                name,
            });
            Ok(sites.len() - 1)
        };
        let mut units = Vec::new();
        for (u, &d) in dilations.iter().enumerate() {
            let alpha = |i: usize| ar.f32(&format!("{p}.alpha{i}.{u}"));
            let alpha = [alpha(1)?, alpha(2)?];
            units.push(Unit {
                conv1: Layer::load(ar, &format!("{p}.convs1.{u}"), 1, d * (kernel - 1) / 2, d, false)?,
                conv2: Layer::load(ar, &format!("{p}.convs2.{u}"), 1, (kernel - 1) / 2, 1, false)?,
                site1: site(format!("{p}.adain1.{u}"))?,
                site2: site(format!("{p}.adain2.{u}"))?,
                inv_alpha: [alpha[0].iter().map(|a| 1.0 / a).collect(), alpha[1].iter().map(|a| 1.0 / a).collect()],
                alpha,
            });
        }
        Ok(ResBlock { units })
    }

    fn state(&self) -> Vec<UnitState> {
        self.units.iter().map(|u| UnitState {
            conv1: ConvState::new(u.conv1.core()),
            conv2: ConvState::new(u.conv2.core()),
            skip: Cols::new(u.conv1.cin(), 0),
        }).collect()
    }

    fn run(&self, be: Backend, aff: &[Affine], st: &mut [UnitState], x: Act, fin: bool) -> Act {
        let mut v = x;
        for (u, s) in self.units.iter().zip(st) {
            s.skip.append(&v);
            let a = affine_snake(&v, &aff[u.site1], &u.alpha[0], &u.inv_alpha[0]);
            let t = u.conv1.run(be, &mut s.conv1, &a, fin);
            let b = affine_snake(&t, &aff[u.site2], &u.alpha[1], &u.inv_alpha[1]);
            let mut y = u.conv2.run(be, &mut s.conv2, &b, fin);
            y.add(&s.skip.pop(y.t));
            v = y;
        }
        v
    }
}

pub struct Generator {
    sites: Vec<Site>,
    noise_convs: Vec<Conv>,
    noise_res: Vec<ResBlock>,
    ups: Vec<Layer>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv,
    window: Vec<f32>,
    hop: usize,
    eps: f64,
}

fn ints(s: &str) -> Result<Vec<usize>> {
    s.split(',').map(|v| v.trim().parse().map_err(|e| format!("{s}: {e}"))).collect()
}

fn leaky(slope: f32) -> impl Fn(f32) -> f32 {
    move |v| if v > 0.0 { v } else { v * slope }
}

impl Generator {
    pub fn load(ar: &Part) -> Result<Self> {
        if ar.meta("kind")? != "tts" {
            return Err("not a Zero Talk model file".into());
        }
        let (rates, kernels) = (ints(ar.meta("upsample_rates")?)?, ints(ar.meta("upsample_kernel_sizes")?)?);
        let dil = ints(ar.meta("resblock_dilations")?)?;
        let n_fft: usize = ar.meta_as("istft_n_fft")?;
        let hop: usize = ar.meta_as("istft_hop")?;
        let eps: f64 = ar.meta_as("norm_eps")?;
        if rates != [10, 6] || kernels != [20, 12] || dil != [1, 3, 5] || n_fft != 20 || hop != 5 {
            return Err("unsupported generator configuration".into());
        }
        let mut sites = Vec::new();
        let (mut noise_convs, mut noise_res, mut ups, mut resblocks) = (vec![], vec![], vec![], vec![]);
        for i in 0..2 {
            let p = format!("noise_convs.{i}");
            let (w, b) = (ar.f32(&format!("{p}.weight"))?, Some(ar.f32(&format!("{p}.bias"))?));
            let shape = ar.shape(&format!("{p}.weight"))?;
            noise_convs.push(if i == 0 {
                let s = rates[1]; // the source spectrum runs at the final rate
                Conv::new(w, shape, b, s, (s + 1) / 2, 1, false)
            } else {
                Conv::new(w, shape, b, 1, 0, 1, false)
            });
            noise_res.push(ResBlock::load(ar, &format!("noise_res.{i}"), [7, 11][i], &dil, &mut sites)?);
            ups.push(Layer::load(ar, &format!("ups.{i}"), rates[i], (kernels[i] - rates[i]) / 2, 1, true)?);
            for (j, k) in [3, 7, 11].into_iter().enumerate() {
                resblocks.push(ResBlock::load(ar, &format!("resblocks.{}", 3 * i + j), k, &dil, &mut sites)?);
            }
        }
        let conv_post = Conv::new(ar.f32("conv_post.weight")?, ar.shape("conv_post.weight")?,
                                  Some(ar.f32("conv_post.bias")?), 1, 3, 1, false);
        Ok(Generator {
            sites,
            noise_convs,
            noise_res,
            ups,
            resblocks,
            conv_post,
            window: ar.f32("stft.window")?,
            hop,
            eps,
        })
    }

    /// Names of the 48 static-norm sites, in the order `condition` expects their moments.
    pub fn sites(&self) -> impl Iterator<Item = &str> {
        self.sites.iter().map(|s| s.name.as_str())
    }

    /// Per-site affines for one utterance from the sentence's length coordinates and each site's static norm (mean, variance):
    /// (1 + gamma) * ((x - mean) / sqrt(var + eps) * w + b) + beta.
    pub fn condition(&self, coords: &[f32; COORDS], moments: &[(Vec<f32>, Vec<f32>)]) -> Result<Vec<Affine>> {
        if moments.len() != self.sites.len() {
            return Err(format!("expected {} norm sites, got {}", self.sites.len(), moments.len()));
        }
        Ok(self.sites.iter().zip(moments).map(|(s, (mean, var))| {
            let h = fold_combine(&s.fold, coords);
            let c = s.norm_w.len();
            let (mut scale, mut shift) = (vec![0.0; c], vec![0.0; c]);
            for i in 0..c {
                let g = 1.0 + h[i];
                scale[i] = (g * s.norm_w[i] as f64 / (var[i] as f64 + self.eps).sqrt()) as f32;
                shift[i] = (g * s.norm_b[i] as f64 + h[c + i]) as f32;
            }
            Affine { mean: mean.clone(), scale, shift }
        }).collect())
    }

    /// A stream for one utterance's conditioning.
    pub fn stream(&self, be: Backend, aff: Vec<Affine>) -> Stream<'_> {
        let states = |blocks: &[ResBlock]| blocks.iter().map(|b| b.state()).collect::<Vec<_>>();
        let chans = |i: usize| self.noise_convs[i].cout;
        Stream {
            g: self,
            be,
            aff,
            noise_conv0: ConvState::new(&self.noise_convs[0]),
            noise_res: states(&self.noise_res),
            ups: self.ups.iter().map(|l| ConvState::new(l.core())).collect(),
            resblocks: states(&self.resblocks),
            post: ConvState::new(&self.conv_post),
            istft: Istft::new(&self.window, self.hop),
            source: (0..2).map(|i| Cols::new(chans(i), 0)).collect(),
            upsampled: (0..2).map(|i| Cols::new(chans(i), 0)).collect(),
            branches: (0..6).map(|j| Cols::new(chans(j / 3), 0)).collect(),
            reflect: Some(Act::zeros(chans(1), 0)),
        }
    }

}

/// The generator's stream state. Push latent frames and source-spectrum columns in any amounts; each push returns
/// the audio that became final, and `finish` the rest.
pub struct Stream<'g> {
    g: &'g Generator,
    be: Backend,
    aff: Vec<Affine>,
    noise_conv0: ConvState,
    noise_res: Vec<Vec<UnitState>>,
    ups: Vec<ConvState>,
    resblocks: Vec<Vec<UnitState>>,
    post: ConvState,
    istft: Istft,
    /// Per upsampling step: the source branch's and the upsampled branch's columns waiting for each other.
    source: Vec<Cols>,
    upsampled: Vec<Cols>,
    /// Resblock outputs waiting for the slowest of their step's three branches.
    branches: Vec<Cols>,
    /// ups.1 columns held until the left reflection's source column (column 1) exists; None once started.
    reflect: Option<Act>,
}

/// Pushes are processed in blocks of at most this many frames: buffers are sized by the block, so memory stays
/// bounded for any push.
const BLOCK: usize = 16;

impl Stream<'_> {
    pub fn push(&mut self, x: &Act, har: &Act) -> Vec<f32> {
        let blocks = x.t.div_ceil(BLOCK).max(har.t.div_ceil(COLUMNS_PER_FRAME * BLOCK));
        if blocks <= 1 {
            return self.run(x.clone(), har, false);
        }
        let mut audio = Vec::new();
        for b in 0..blocks {
            let last = b + 1 == blocks;
            let cut = |a: &Act, size: usize| {
                let (lo, hi) = ((b * size).min(a.t), if last { a.t } else { ((b + 1) * size).min(a.t) });
                a.columns(lo, hi)
            };
            audio.extend(self.run(cut(x, BLOCK), &cut(har, COLUMNS_PER_FRAME * BLOCK), false));
        }
        audio
    }

    pub fn finish(&mut self) -> Vec<f32> {
        self.run(Act::zeros(512, 0), &Act::zeros(22, 0), true)
    }

    fn run(&mut self, mut x: Act, har: &Act, fin: bool) -> Vec<f32> {
        let (g, be) = (self.g, self.be);
        // Source branches.
        let s0 = self.noise_conv0.run(be, &g.noise_convs[0], har, fin);
        self.source[0].append(&g.noise_res[0].run(be, &self.aff, &mut self.noise_res[0], s0, fin));
        let s1 = pointwise(be, &g.noise_convs[1], har);
        self.source[1].append(&g.noise_res[1].run(be, &self.aff, &mut self.noise_res[1], s1, fin));
        // Latent path.
        x.map(leaky(0.1));
        let u0 = g.ups[0].run(be, &mut self.ups[0], &x, fin);
        self.upsampled[0].append(&u0);
        let mut z = self.step(0, fin);
        z.map(leaky(0.1));
        let u1 = g.ups[1].run(be, &mut self.ups[1], &z, fin);
        let u1 = self.reflect_left(u1, fin);
        self.upsampled[1].append(&u1);
        let mut z = self.step(1, fin);
        z.map(leaky(0.01));
        let post = self.post.run(be, &g.conv_post, &z, fin);
        self.istft.run(&post, fin)
    }

    /// x = upsampled + source, then the mean of the step's three resblocks.
    fn step(&mut self, i: usize, fin: bool) -> Act {
        let m = self.upsampled[i].len().min(self.source[i].len());
        assert!(!fin || self.upsampled[i].len() == self.source[i].len(), "upsampling step {i} branches differ in length");
        let mut y = self.upsampled[i].pop(m);
        y.add(&self.source[i].pop(m));
        for j in 3 * i..3 * i + 3 {
            // The last resblock takes y itself, the others a copy.
            let input = if j + 1 == 3 * i + 3 { std::mem::take(&mut y) } else { y.clone() };
            let r = self.g.resblocks[j].run(self.be, &self.aff, &mut self.resblocks[j], input, fin);
            self.branches[j].append(&r);
        }
        let n = (3 * i..3 * i + 3).map(|j| self.branches[j].len()).min().unwrap();
        let mut z = self.branches[3 * i].pop(n);
        z.add(&self.branches[3 * i + 1].pop(n));
        z.add(&self.branches[3 * i + 2].pop(n));
        z.map(|v| v / 3.0);
        z
    }

    /// ReflectionPad1d((1, 0)): the stream starts with column 1, then all columns.
    fn reflect_left(&mut self, u: Act, fin: bool) -> Act {
        let Some(held) = self.reflect.take() else { return u };
        let h = held.cat(&u);
        if h.t < 2 {
            assert!(!fin, "too short for the reflection pad");
            let c = h.c;
            self.reflect = Some(h);
            return Act::zeros(c, 0);
        }
        h.columns(1, 2).cat(&h)
    }
}
