//! The harmonic source and its STFT.
//!
//! F0 at 80 Hz becomes nine harmonics whose phase is accumulated in float64 at the frame rate, rounded to float32 knots and
//! interpolated linearly to 24 kHz; voiced samples carry the sines plus a little noise, unvoiced samples noise only; a 9-to-1 linear
//! layer and tanh give the waveform. The phase that reaches `sin` is of the order of 10^5 rad on a long utterance, where float32
//! spacing is 0.016 rad, so every float32 and float64 step is taken in the same order on every run.
//!
//! The noise is torch's CPU generator: mt19937 seeded with the seed, nine draws consumed up front (torch.rand(1, 9)), then
//! torch.randn per group of four frames, filled with Box-Muller in blocks of sixteen like ATen's `normal_fill`. A render is
//! therefore reproducible by seed.
//!
//! Based on kokoro 0.9.4's SineGen and SourceModuleHnNSF (Apache-2.0), and changed for this engine.
use crate::archive::{Part, Result};
use crate::kernels::{dft_angles, Act};
use crate::pipeline::SAMPLE_RATE;

/// The standard 32-bit Mersenne Twister.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut s = [0u32; 624];
        s[0] = seed;
        for i in 1..624 {
            s[i] = 1_812_433_253u32.wrapping_mul(s[i - 1] ^ (s[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { state: s, index: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let x = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7FFF_FFFF);
                let mut v = x >> 1;
                if x & 1 != 0 {
                    v ^= 0x9908_B0DF;
                }
                self.state[i] = self.state[(i + 397) % 624] ^ v;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^ (y >> 18)
    }

    /// torch's float uniform in [0, 1): the low 24 bits of a draw.
    fn uniform(&mut self) -> f32 {
        (self.next_u32() & 0xFF_FFFF) as f32 * (1.0 / 16_777_216.0)
    }

    /// `torch.randn(n)` on the CPU for n >= 16: uniforms for every element, then Box-Muller over blocks of 16; when n is not
    /// a multiple of 16 the last 16 values are redrawn and transformed again.
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        assert!(n >= 16);
        let mut data: Vec<f32> = (0..n).map(|_| self.uniform()).collect();
        for i in (0..=n - 16).step_by(16) {
            box_muller_16(&mut data[i..i + 16]);
        }
        if n % 16 != 0 {
            for v in &mut data[n - 16..] {
                *v = self.uniform();
            }
            box_muller_16(&mut data[n - 16..]);
        }
        data
    }
}

fn box_muller_16(d: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0 - d[j];
        let u2 = d[j + 8];
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = (2.0f32 as f64 * std::f64::consts::PI * u2 as f64) as f32;
        d[j] = radius * theta.cos();
        d[j + 8] = radius * theta.sin();
    }
}

/// The seed of every render.
pub const SEED: u32 = 1729;
const DIM: usize = 9; // the fundamental and eight overtones
const RATE: f32 = SAMPLE_RATE as f32;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
pub const FRAME: usize = 300; // samples per 80 Hz frame
/// Spectrum columns per 80 Hz frame: the STFT in pipeline.rs hops 5 samples.
pub const COLUMNS_PER_FRAME: usize = 60;
const QUANTUM: usize = 4; // frames per group
const WAVE_PIECE_FRAMES: usize = 16 * QUANTUM; // the frames per push of `Source::wave`

/// The source's only weights: the 9-to-1 linear layer.
pub struct Source {
    w: [f32; DIM],
    b: f32,
}

impl Source {
    pub fn load(ar: &Part) -> Result<Self> {
        let w = ar.f32("m_source.l_linear.weight")?;
        let b = *ar.f32("m_source.l_linear.bias")?.first().ok_or("m_source.l_linear.bias: expected 1 value")?;
        Ok(Source { w: w.try_into().map_err(|_| "m_source.l_linear.weight: expected 9 weights")?, b })
    }

    /// The whole sentence's source wave for F0 `f0_hz` (Hz per 80 Hz frame), its random phase seeded `seed`: what a stream gives for all the
    /// frames pushed at once, then finished. The F0 goes in by pieces of whole groups (any split of the pushes gives the same wave) into one
    /// array of the final size, so the wave is never held twice.
    pub fn wave(&self, seed: u32, f0_hz: &[f32]) -> Vec<f32> {
        let mut stream = self.stream(seed);
        let mut wave = Vec::with_capacity(FRAME * f0_hz.len());
        for piece in f0_hz.chunks(WAVE_PIECE_FRAMES) {
            wave.extend(stream.push(piece));
        }
        wave.extend(stream.finish());
        wave
    }

    pub fn stream(&self, seed: u32) -> SourceStream<'_> {
        let mut rng = Mt19937::new(seed);
        for _ in 0..DIM {
            rng.next_u32(); // torch.rand(1, 9), whose values are not used
        }
        SourceStream { src: self, rng, f0: Vec::new(), phase_sum: [0.0; DIM], previous_knot: None, frames: 0 }
    }
}

pub struct SourceStream<'a> {
    src: &'a Source,
    rng: Mt19937,
    f0: Vec<f32>,
    phase_sum: [f64; DIM],
    previous_knot: Option<[f32; DIM]>,
    frames: usize,
}

impl SourceStream<'_> {
    /// F0 (Hz, one value per 80 Hz frame) in; the source samples of every complete group of four frames out. A group needs
    /// one frame beyond it for interpolation, so up to four frames wait for the next push or `finish`.
    pub fn push(&mut self, f0: &[f32]) -> Vec<f32> {
        self.f0.extend_from_slice(f0);
        let mut out = Vec::new();
        while self.f0.len() > QUANTUM {
            out.extend(self.group(QUANTUM, false));
        }
        out
    }

    /// The samples of the frames still waiting; the last knot is held, as in the offline interpolation.
    pub fn finish(&mut self) -> Vec<f32> {
        let n = self.f0.len();
        if n == 0 { Vec::new() } else { self.group(n, true) }
    }

    fn group(&mut self, count: usize, last: bool) -> Vec<f32> {
        let take = (count + 1).min(self.f0.len());
        // Phase increments per frame, accumulated in float64 (cumulative sum first, then the carried phase), then float32 knots.
        let two_pi = (2.0 * std::f64::consts::PI) as f32;
        let (carried, mut run, mut next) = (self.phase_sum, [0f64; DIM], self.phase_sum);
        let mut knots: Vec<[f32; DIM]> = Vec::with_capacity(take + 1);
        if let Some(k) = self.previous_knot {
            knots.push(k);
        }
        let first = knots.len();
        for t in 0..take {
            let mut knot = [0f32; DIM];
            for h in 0..DIM {
                run[h] += (((self.f0[t] * (h + 1) as f32) / RATE) % 1.0) as f64;
                let sum = run[h] + carried[h];
                if t == count - 1 {
                    next[h] = sum;
                }
                knot[h] = ((sum as f32) * two_pi) * FRAME as f32;
            }
            knots.push(knot);
        }
        self.phase_sum = next;
        let base = self.frames;
        let knot_base = if first == 1 { base as i64 - 1 } else { 0 };
        self.previous_knot = Some(knots[count - 1 + first]);

        let scale = (1.0f32 / FRAME as f32) as f64; // float(torch.tensor(1 / 300))
        let n = count * FRAME;
        let noise = self.rng.randn(n * DIM);
        let voiced: Vec<f32> = self.f0[..count].iter().map(|&f| if f > VOICED_THRESHOLD { 1.0 } else { 0.0 }).collect();
        let mut wave = Vec::with_capacity(n);
        for s in 0..n {
            let idx = base * FRAME + s;
            let coordinate = (((idx as f64 + 0.5) * scale - 0.5).max(0.0)) as f32;
            let mut left = coordinate as i64;
            let mut right = left + 1;
            if last {
                let top = (base + count) as i64 - 1;
                left = left.min(top);
                right = right.min(top);
            }
            let weight = (coordinate - left as f32).clamp(0.0, 1.0);
            let (a, b) = (&knots[(left - knot_base) as usize], &knots[(right - knot_base) as usize]);
            let uv = voiced[s / FRAME];
            let noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
            let mut acc = 0f32;
            for h in 0..DIM {
                let phase = ((a[h] as f64) * ((1.0 - weight) as f64) + ((b[h] * weight) as f64)) as f32;
                let x = phase.sin() * SINE_AMP * uv + noise_amp * noise[s * DIM + h];
                acc += self.src.w[h] * x;
            }
            wave.push((acc + self.src.b).tanh());
        }
        self.f0.drain(..count);
        self.frames += count;
        wave
    }
}

/// `torch.stft(x, n_fft, hop, window, center=True)` (reflection padding) as the generator consumes it: 11 magnitude rows
/// then 11 angle rows, one column per hop, computed in float64 from the float32 samples.
pub struct Stft {
    window: Vec<f64>,
    hop: usize,
}

impl Stft {
    pub fn new(window: &[f32], hop: usize) -> Self {
        Stft { window: window.iter().map(|&w| w as f64).collect(), hop }
    }

    pub fn transform(&self, wave: &[f32]) -> Act {
        let (n_fft, frames) = (self.window.len(), 1 + wave.len() / self.hop);
        let bins = n_fft / 2 + 1;
        let (cos, sin) = dft_angles(n_fft);
        let mut out = Act::zeros(2 * bins, frames);
        let mut seg = vec![0f64; n_fft];
        // Each column's bins go straight to the output (the same values as `complex_planes` gives, without holding its two float64 planes).
        for f in 0..frames {
            self.segment(wave, f, &mut seg);
            for k in 0..bins {
                let (re, im) = Self::bin(&seg, &cos, &sin, k, bins);
                out.data[k * frames + f] = re.hypot(im) as f32;
                out.data[bins * frames + k * frames + f] = im.atan2(re) as f32;
            }
        }
        out
    }

    /// The window times the reflection-padded samples of column `f`.
    fn segment(&self, wave: &[f32], f: usize, seg: &mut [f64]) {
        let (n_fft, n) = (self.window.len(), wave.len());
        let at = |i: isize| -> f64 {
            let mut j = i - (n_fft / 2) as isize;
            if j < 0 {
                j = -j;
            }
            if j >= n as isize {
                j = 2 * (n as isize - 1) - j;
            }
            wave[j as usize] as f64
        };
        for (j, v) in seg.iter_mut().enumerate() {
            *v = at((f * self.hop + j) as isize) * self.window[j];
        }
    }

    /// Bin `k` of a segment: its real and imaginary parts.
    fn bin(seg: &[f64], cos: &[f64], sin: &[f64], k: usize, bins: usize) -> (f64, f64) {
        let n_fft = seg.len();
        let (mut re, mut im) = (0f64, 0f64);
        for j in 0..n_fft {
            re += seg[j] * cos[k * n_fft + j];
            im -= seg[j] * sin[k * n_fft + j];
        }
        if k == 0 || k == bins - 1 {
            im = 0.0; // a real signal's DC and Nyquist bins are real, as in torch's real FFT: the angle is 0 or +pi
        }
        (re, im)
    }

    /// The real and imaginary parts, in float64, that `transform` takes the magnitude and the angle of: `bins` rows of `frames` columns
    /// each, and the number of columns. The angle of a bin of almost no energy is decided by the sign of its imaginary part, so a test
    /// of the angles needs these.
    pub fn complex_planes(&self, wave: &[f32]) -> (Vec<f64>, Vec<f64>, usize) {
        let (n_fft, frames) = (self.window.len(), 1 + wave.len() / self.hop);
        let bins = n_fft / 2 + 1;
        let (cos, sin) = dft_angles(n_fft);
        let (mut re_plane, mut im_plane) = (vec![0f64; bins * frames], vec![0f64; bins * frames]);
        let mut seg = vec![0f64; n_fft];
        for f in 0..frames {
            self.segment(wave, f, &mut seg);
            for k in 0..bins {
                let (re, im) = Self::bin(&seg, &cos, &sin, k, bins);
                re_plane[k * frames + f] = re;
                im_plane[k * frames + f] = im;
            }
        }
        (re_plane, im_plane, frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_twister_reference_outputs() {
        // The first outputs of the Mersenne Twister for its default seed 5489.
        let mut r = Mt19937::new(5489);
        assert_eq!([r.next_u32(), r.next_u32(), r.next_u32()], [3_499_211_612, 581_869_302, 3_890_346_734]);
    }
}
