//! Compute kernels: one GEMM interface with two backends, convolutions issued as GEMMs together with their streaming
//! state, the fused per-channel affine + Snake, and the streaming iSTFT. All arithmetic is float32.

/// GEMM backend. `Portable` (matrixmultiply: runtime-detected AVX2/FMA, AVX-512 or NEON) builds everywhere;
/// `Accelerate` exists with the `accelerate` feature on macOS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backend {
    Portable,
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    Accelerate,
}

impl Backend {
    /// The backends compiled into this build, the fastest first: that one is the default ([`Backend::fastest`]).
    pub fn available() -> Vec<Backend> {
        #[allow(unused_mut)]
        let mut all = vec![Backend::Portable];
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        all.insert(0, Backend::Accelerate);
        all
    }

    /// The default: Accelerate on macOS, else the portable backend.
    pub fn fastest() -> Backend {
        Self::available()[0]
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "portable" => Some(Backend::Portable),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            "accelerate" => Some(Backend::Accelerate),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Portable => "portable",
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Backend::Accelerate => "accelerate",
        }
    }
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32,
                   b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32);
}

/// C = A B + beta C for row-major A (m x k), B (k x n), C (m x n) with row strides lda, ldb, ldc.
/// With beta = 0, C is not read. Safety: the pointers must cover those extents.
pub unsafe fn gemm(be: Backend, m: usize, n: usize, k: usize, a: *const f32, lda: usize, b: *const f32, ldb: usize,
                   beta: f32, c: *mut f32, ldc: usize) {
    match be {
        Backend::Portable => matrixmultiply::sgemm(m, k, n, 1.0, a, lda as isize, 1, b, ldb as isize, 1, beta, c,
                                                   ldc as isize, 1),
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        Backend::Accelerate => cblas_sgemm(101, 111, 111, m as i32, n as i32, k as i32, 1.0, a, lda as i32, b,
                                           ldb as i32, beta, c, ldc as i32),
    }
}

/// Activations [c][t], row-major.
#[derive(Clone, Default)]
pub struct Act {
    pub c: usize,
    pub t: usize,
    pub data: Vec<f32>,
}

impl Act {
    pub fn zeros(c: usize, t: usize) -> Self {
        Act { c, t, data: vec![0.0; c * t] }
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.t..(i + 1) * self.t]
    }
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.t..(i + 1) * self.t]
    }
    /// Columns [a, b).
    pub fn columns(&self, a: usize, b: usize) -> Act {
        Act { c: self.c, t: b - a, data: (0..self.c).flat_map(|i| self.row(i)[a..b].iter().copied()).collect() }
    }
    /// Rows [a, b).
    pub fn rows(&self, a: usize, b: usize) -> Act {
        Act { c: b - a, t: self.t, data: self.data[a * self.t..b * self.t].to_vec() }
    }
    /// Columns of self followed by those of `other`.
    pub fn cat(self, other: &Act) -> Act {
        if other.t == 0 {
            return self;
        }
        if self.t == 0 {
            return other.clone();
        }
        assert_eq!(self.c, other.c);
        let t = self.t + other.t;
        Act { c: self.c, t, data: (0..self.c).flat_map(|i| self.row(i).iter().chain(other.row(i)).copied()).collect() }
    }
    pub fn add(&mut self, other: &Act) {
        assert!(self.c == other.c && self.t == other.t);
        self.data.iter_mut().zip(&other.data).for_each(|(y, x)| *y += *x);
    }
    pub fn map(&mut self, f: impl Fn(f32) -> f32) {
        self.data.iter_mut().for_each(|y| *y = f(*y));
    }
}

/// Columns [start, start + len) of a column stream, kept between pushes: convolution histories and alignment queues.
/// `start` may be negative (leading zero padding). Row r's columns live at data[r * cap + off ..][..len].
pub struct Cols {
    c: usize,
    cap: usize,
    off: usize,
    len: usize,
    start: isize,
    data: Vec<f32>,
}

impl Cols {
    pub fn new(c: usize, start: isize) -> Self {
        Cols { c, cap: 0, off: 0, len: 0, start, data: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn end(&self) -> isize {
        self.start + self.len as isize
    }
    /// Room for n more columns: compact in place, or grow to fit exactly (pushes are bounded, so sizes settle).
    fn reserve(&mut self, n: usize) {
        if self.off + self.len + n <= self.cap {
            return;
        }
        let cap = self.cap.max(self.len + n);
        if cap == self.cap {
            for r in 0..self.c {
                let b = r * cap;
                self.data.copy_within(b + self.off..b + self.off + self.len, b);
            }
        } else {
            let mut data = vec![0.0; self.c * cap];
            for r in 0..self.c {
                let src = r * self.cap + self.off;
                data[r * cap..r * cap + self.len].copy_from_slice(&self.data[src..src + self.len]);
            }
            self.data = data;
            self.cap = cap;
        }
        self.off = 0;
    }
    pub fn append(&mut self, a: &Act) {
        if a.t == 0 {
            return;
        }
        assert_eq!(a.c, self.c);
        self.reserve(a.t);
        for r in 0..self.c {
            let d = r * self.cap + self.off + self.len;
            self.data[d..d + a.t].copy_from_slice(a.row(r));
        }
        self.len += a.t;
    }
    pub fn append_zeros(&mut self, n: usize) {
        self.reserve(n);
        for r in 0..self.c {
            let d = r * self.cap + self.off + self.len;
            self.data[d..d + n].fill(0.0);
        }
        self.len += n;
    }
    /// Removes and returns the first n columns.
    pub fn pop(&mut self, n: usize) -> Act {
        assert!(n <= self.len);
        let mut a = Act::zeros(self.c, n);
        for r in 0..self.c {
            let s = r * self.cap + self.off;
            a.row_mut(r).copy_from_slice(&self.data[s..s + n]);
        }
        self.discard_to(self.start + n as isize);
        a
    }
    /// Drops the columns before `pos`.
    pub fn discard_to(&mut self, pos: isize) {
        let n = (pos - self.start).clamp(0, self.len as isize) as usize;
        self.off += n;
        self.len -= n;
        self.start += n as isize;
    }
    /// Row r, columns [pos, pos + n), which must be held.
    pub fn span_mut(&mut self, r: usize, pos: isize, n: usize) -> &mut [f32] {
        assert!(pos >= self.start && pos + n as isize <= self.end());
        let s = r * self.cap + self.off + (pos - self.start) as usize;
        &mut self.data[s..s + n]
    }
    /// Pointer to row 0 at column `pos` (row stride `ld()`), for GEMMs.
    fn ptr(&self, pos: isize) -> *const f32 {
        assert!(pos >= self.start && pos < self.end());
        self.data[self.off + (pos - self.start) as usize..].as_ptr()
    }
    fn ld(&self) -> usize {
        self.cap
    }
}

/// A 1-D convolution (or transposed convolution) as GEMMs.
pub struct Conv {
    pub cout: usize,
    pub cin: usize,
    pub k: usize,
    stride: usize,
    pad: usize,
    dil: usize,
    transposed: bool,
    /// Per-tap matrices [k][cout][cin], or for strided convolutions [cout][cin * k] (im2col order).
    w: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Conv {
    /// `weight` in PyTorch layout: Conv1d [cout][cin][k], ConvTranspose1d [cin][cout][k].
    pub fn new(weight: Vec<f32>, shape: &[usize], bias: Option<Vec<f32>>, stride: usize, pad: usize, dil: usize,
               transposed: bool) -> Self {
        let (cout, cin, k) = if transposed { (shape[1], shape[0], shape[2]) } else { (shape[0], shape[1], shape[2]) };
        assert_eq!(weight.len(), cout * cin * k);
        assert!(!transposed || dil == 1);
        let w = if stride > 1 && !transposed {
            weight
        } else {
            let mut w = vec![0.0; weight.len()];
            for o in 0..cout {
                for i in 0..cin {
                    for j in 0..k {
                        let src = if transposed { (i * cout + o) * k + j } else { (o * cin + i) * k + j };
                        w[(j * cout + o) * cin + i] = weight[src];
                    }
                }
            }
            w
        };
        if let Some(b) = &bias {
            assert_eq!(b.len(), cout);
        }
        Conv { cout, cin, k, stride, pad, dil, transposed, w, bias }
    }

    pub fn out_len(&self, t: usize) -> usize {
        if t == 0 {
            0
        } else if self.transposed {
            (t - 1) * self.stride + self.k - 2 * self.pad
        } else {
            (t + 2 * self.pad - self.dil * (self.k - 1) - 1) / self.stride + 1
        }
    }

    fn pointwise(&self) -> bool {
        self.k == 1 && self.stride == 1 && !self.transposed
    }

    /// Direct convolution of y.t outputs: output o, tap j reads input column o * stride + j * dil of x (row stride ldx).
    /// Safety: x must hold cin rows and every column read.
    unsafe fn run(&self, be: Backend, x: *const f32, ldx: usize, y: &mut Act) {
        let n = y.t;
        if n == 0 {
            return;
        }
        let mut beta = 0.0;
        if let Some(b) = &self.bias {
            (0..self.cout).for_each(|o| y.row_mut(o).fill(b[o]));
            beta = 1.0;
        }
        let yp = y.data.as_mut_ptr();
        if self.stride == 1 {
            // Shifted GEMMs, one per tap.
            let cc = self.cout * self.cin;
            for j in 0..self.k {
                gemm(be, self.cout, n, self.cin, self.w[j * cc..].as_ptr(), self.cin, x.add(j * self.dil), ldx, beta,
                     yp, n);
                beta = 1.0;
            }
        } else {
            // im2col: col[(i k + j)][o] = x[i][o stride + j dil].
            let kk = self.cin * self.k;
            let mut col = vec![0.0f32; kk * n];
            for i in 0..self.cin {
                for j in 0..self.k {
                    let src = x.add(i * ldx + j * self.dil);
                    for (o, d) in col[(i * self.k + j) * n..][..n].iter_mut().enumerate() {
                        *d = *src.add(o * self.stride);
                    }
                }
            }
            gemm(be, self.cout, n, kk, self.w.as_ptr(), kk, col.as_ptr(), n, beta, yp, n);
        }
    }
}

/// A stateless 1x1 convolution.
pub fn pointwise(be: Backend, c: &Conv, x: &Act) -> Act {
    assert!(c.pointwise() && x.c == c.cin);
    let mut y = Act::zeros(c.cout, x.t);
    unsafe { c.run(be, x.data.as_ptr(), x.t, &mut y) };
    y
}

/// Streaming state of one convolution. Direct convolutions keep the input columns their next outputs read, starting
/// with `pad` zero columns (the offline zero padding); transposed ones keep the output columns that still receive
/// contributions. Pointwise convolutions keep nothing.
pub struct ConvState {
    cols: Cols,
    n_in: usize,
    n_out: usize,
}

impl ConvState {
    pub fn new(c: &Conv) -> Self {
        let cols = if c.transposed {
            Cols::new(c.cout, 0)
        } else {
            let mut cols = Cols::new(c.cin, -(c.pad as isize));
            cols.append_zeros(c.pad);
            cols
        };
        ConvState { cols, n_in: 0, n_out: 0 }
    }

    /// Takes input columns and returns the output columns that became final; with `fin` the input has ended and all
    /// remaining outputs are returned (zero padding on the right, as offline).
    pub fn run(&mut self, be: Backend, c: &Conv, x: &Act, fin: bool) -> Act {
        if c.pointwise() {
            pointwise(be, c, x)
        } else if c.transposed {
            self.transposed(be, c, x, fin)
        } else {
            self.direct(be, c, x, fin)
        }
    }

    fn direct(&mut self, be: Backend, c: &Conv, x: &Act, fin: bool) -> Act {
        self.cols.append(x);
        self.n_in += x.t;
        let (s, p, span) = (c.stride as isize, c.pad as isize, (c.dil * (c.k - 1)) as isize);
        let avail = if fin {
            let total = c.out_len(self.n_in);
            let end = (total as isize - 1) * s - p + span + 1; // past the last input column read
            if end > self.cols.end() {
                self.cols.append_zeros((end - self.cols.end()) as usize);
            }
            total
        } else {
            let last = self.n_in as isize - 1 + p - span; // the last output's first tap
            if last < 0 { 0 } else { last as usize / c.stride + 1 }
        };
        let mut y = Act::zeros(c.cout, avail - self.n_out);
        if y.t > 0 {
            unsafe { c.run(be, self.cols.ptr(self.n_out as isize * s - p), self.cols.ld(), &mut y) };
        }
        self.n_out = avail;
        self.cols.discard_to(avail as isize * s - p);
        y
    }

    fn transposed(&mut self, be: Backend, c: &Conv, x: &Act, fin: bool) -> Act {
        let (s, p) = (c.stride, c.pad as isize);
        if x.t > 0 {
            // Input column i, tap j adds to output i * stride + j - pad.
            let i0 = self.n_in;
            self.n_in += x.t;
            let end = ((self.n_in - 1) * s + c.k) as isize - p;
            if end > self.cols.end() {
                self.cols.append_zeros((end - self.cols.end()) as usize);
            }
            let (cc, m) = (c.cout * c.cin, x.t);
            let mut tmp = vec![0.0f32; c.cout * m];
            for j in 0..c.k {
                unsafe { gemm(be, c.cout, m, c.cin, c.w[j * cc..].as_ptr(), c.cin, x.data.as_ptr(), m, 0.0,
                              tmp.as_mut_ptr(), m) };
                let first = (i0 * s + j) as isize - p;
                let skip = (-first).max(0) as usize; // outputs before 0 are cropped
                let lo = skip.div_ceil(s);
                if lo >= m {
                    continue;
                }
                let pos = first + (lo * s) as isize;
                for o in 0..c.cout {
                    let row = self.cols.span_mut(o, pos, (m - 1 - lo) * s + 1);
                    for (i, v) in tmp[o * m + lo..(o + 1) * m].iter().enumerate() {
                        row[i * s] += *v;
                    }
                }
            }
        }
        let avail = if fin { c.out_len(self.n_in) } else { (self.n_in * s).saturating_sub(c.pad) };
        let mut y = self.cols.pop(avail - self.n_out);
        if let Some(b) = &c.bias {
            (0..c.cout).for_each(|o| y.row_mut(o).iter_mut().for_each(|v| *v += b[o]));
        }
        self.n_out = avail;
        y
    }
}

/// Per-channel affine (static norm and AdaIN folded): y = (x - mean) * scale + shift.
#[derive(Clone, Debug)]
pub struct Affine {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    pub shift: Vec<f32>,
}

/// Snake(affine(x)) = v + sin(alpha v)^2 / alpha with v the affine output, in one pass.
pub fn affine_snake(x: &Act, a: &Affine, alpha: &[f32], inv_alpha: &[f32]) -> Act {
    assert_eq!(a.mean.len(), x.c);
    let mut y = Act::zeros(x.c, x.t);
    for c in 0..x.c {
        let (m, k, s, al, ia) = (a.mean[c], a.scale[c], a.shift[c], alpha[c], inv_alpha[c]);
        for (o, &v) in y.row_mut(c).iter_mut().zip(x.row(c)) {
            #[cfg(target_arch = "aarch64")]
            {
                let t = (v - m).mul_add(k, s);
                *o = ia.mul_add(sin2(al * t), t);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let t = (v - m) * k + s;
                *o = t + ia * sin2(al * t);
            }
        }
    }
    y
}

/// sin(v)^2 without branches or calls, so the loop around it vectorizes on any SIMD target. Range reduction by pi
/// (SLEEF's four-part constants for non-FMA arithmetic, exact for |v| < 2^13 pi) and SLEEF's sinf_u35 polynomial;
/// the sign of sin is irrelevant after squaring. Round-to-nearest uses the 1.5 * 2^23 trick (valid for |v| < 2^22 pi).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn sin2(v: f32) -> f32 {
    // Fused multiply-adds, which aarch64 has in hardware (about 10% less generator time); the results differ from the
    // unfused version below in the last bits, within the parity gate.
    const MAGIC: f32 = 12_582_912.0;
    let q = v.mul_add(std::f32::consts::FRAC_1_PI, MAGIC) - MAGIC;
    let mut d = (-q).mul_add(3.140625, v);
    d = (-q).mul_add(0.000_967_025_756_835_937_5, d);
    d = (-q).mul_add(6.277_114_152_908_325_195_3e-7, d);
    d = (-q).mul_add(1.215_420_125_655_342_076_2e-10, d);
    let s = d * d;
    let mut u: f32 = 2.608_315_980_978_659_354_150_3e-6;
    u = u.mul_add(s, -0.000_198_106_907_191_686_332_225_8);
    u = u.mul_add(s, 0.008_333_078_585_565_090_179_443_36);
    u = u.mul_add(s, -0.166_666_597_127_914_428_710_938);
    let r = (s * u).mul_add(d, d);
    r * r
}

/// The same without fused multiply-adds: other targets have no FMA in the baseline instruction set, where `mul_add` would be a
/// slow library call, and WebAssembly keeps its bytes the same everywhere.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn sin2(v: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0;
    let q = (v * std::f32::consts::FRAC_1_PI + MAGIC) - MAGIC;
    let mut d = v - q * 3.140625;
    d -= q * 0.000_967_025_756_835_937_5;
    d -= q * 6.277_114_152_908_325_195_3e-7;
    d -= q * 1.215_420_125_655_342_076_2e-10;
    let s = d * d;
    let mut u = 2.608_315_980_978_659_354_150_3e-6;
    u = u * s - 0.000_198_106_907_191_686_332_225_8;
    u = u * s + 0.008_333_078_585_565_090_179_443_36;
    u = u * s - 0.166_666_597_127_914_428_710_938;
    let r = s * (u * d) + d;
    r * r
}

/// Cosines and sines of the angles 2 pi (k j) / n of an n-point real DFT, for the bins k = 0 .. n/2 and the samples j < n, row-major
/// [bins][n], in float64 (the STFT in source.rs and the inverse transform below both use them).
pub fn dft_angles(n: usize) -> (Vec<f64>, Vec<f64>) {
    let angle = |i: usize| std::f64::consts::TAU * ((i / n) * (i % n)) as f64 / n as f64;
    (0..(n / 2 + 1) * n).map(|i| (angle(i).cos(), angle(i).sin())).unzip()
}

/// Streaming torch.istft(mag * exp(i phase), n_fft, hop, window, center=True), with mag = exp(post[:bins]) and
/// phase = sin(post[bins:]) from conv_post's output columns [2 bins][frames]. Keeps the overlap-add and window
/// envelope of the samples that later frames still reach.
pub struct Istft {
    window: Vec<f32>,
    hop: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    sum: Cols,
    env: Cols,
    frames: usize,
    emitted: usize,
}

impl Istft {
    pub fn new(window: &[f32], hop: usize) -> Self {
        let (cos, sin) = dft_angles(window.len());
        Istft {
            window: window.to_vec(),
            hop,
            cos: cos.iter().map(|&c| c as f32).collect(),
            sin: sin.iter().map(|&s| s as f32).collect(),
            sum: Cols::new(1, 0),
            env: Cols::new(1, 0),
            frames: 0,
            emitted: 0,
        }
    }

    pub fn run(&mut self, post: &Act, fin: bool) -> Vec<f32> {
        let (n_fft, hop) = (self.window.len(), self.hop);
        let bins = n_fft / 2 + 1;
        assert_eq!(post.c, 2 * bins);
        let (mut re, mut im, mut frame) = (vec![0.0f32; bins], vec![0.0f32; bins], vec![0.0f32; n_fft]);
        let norm = 1.0 / n_fft as f32;
        for f in 0..post.t {
            for k in 0..bins {
                let (mag, ph) = (post.row(k)[f].exp(), post.row(bins + k)[f].sin());
                re[k] = mag * ph.cos();
                im[k] = mag * ph.sin();
            }
            for (n, v) in frame.iter_mut().enumerate() {
                // Inverse real DFT; the imaginary parts of the DC and Nyquist bins are ignored, as by irfft.
                let mut acc = re[0] + if n % 2 == 0 { re[bins - 1] } else { -re[bins - 1] };
                for k in 1..bins - 1 {
                    acc += 2.0 * (re[k] * self.cos[k * n_fft + n] - im[k] * self.sin[k * n_fft + n]);
                }
                *v = acc * norm * self.window[n];
            }
            let pos = ((self.frames + f) * hop) as isize;
            let grow = (pos + n_fft as isize - self.sum.end()).max(0) as usize;
            self.sum.append_zeros(grow);
            self.env.append_zeros(grow);
            self.sum.span_mut(0, pos, n_fft).iter_mut().zip(&frame).for_each(|(y, v)| *y += *v);
            self.env.span_mut(0, pos, n_fft).iter_mut().zip(&self.window).for_each(|(e, w)| *e += *w * *w);
        }
        self.frames += post.t;
        // Output sample i is padded position i + n_fft / 2; it is final once no later frame starts at or before it.
        let half = n_fft / 2;
        let end = if fin { hop * self.frames.saturating_sub(1) } else { (hop * self.frames).saturating_sub(half) };
        let n = end.saturating_sub(self.emitted);
        let at = (self.emitted + half) as isize;
        let out = if n == 0 {
            Vec::new()
        } else {
            let y = self.sum.span_mut(0, at, n).to_vec();
            y.iter().zip(self.env.span_mut(0, at, n).iter()).map(|(y, e)| y / e).collect()
        };
        self.emitted += n;
        // Positions before the next frame's start are final (the first n_fft / 2 are trimmed, never emitted).
        let keep = (at + n as isize).min((hop * self.frames) as isize);
        self.sum.discard_to(keep);
        self.env.discard_to(keep);
        out
    }
}
