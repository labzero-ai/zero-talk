// WebGPU kernels for the generator: the tiled convolution with an input prologue, the vectorized transposed convolution,
// the elementwise and polar kernels, and the inverse STFT. The prologue is a static per-channel affine and snake
// (t = (v - mean) * scale + shift; t + invAlpha * sin(alpha t)^2), so there is no statistics pass. The inverse STFT is
// torch.istft with its window-envelope division, in two kernels.
// Float32 throughout. Channel counts are multiples of 4 (and of 8 for the transposed kernel).

export const TILE = { TX: 32, TY: 4, CO_PT: 8, L_PT: 8 };       // K-tap convolutions
export const STRIDED = { TX: 8, TY: 4, CO_PT: 8, L_PT: 8 };     // the stride-6 noise convolution (input tile fits 16 KB)
export const SHARED = 16384;                                    // WebGPU's default workgroup storage limit, in bytes

// Convolution out[co][l] = bias[co] + sum over ci, k of w[ci][k][co] * in[ci][stride l + k dil - pad], where in is the
// source (length lin, zeros outside) or, with `prologue`, t = (v - mean) * scale + shift; t + invAlpha sin(alpha t)^2
// applied to the in-range values only (zero padding comes after the prologue).
// Weights: vec4 array packed [cin][k][cout] with cout a multiple of 4. The prologue buffers are per input channel:
// pro (vec4: mean, scale, shift, alpha) and ia (inverse alpha).
export const convWGSL = ({ k, dil, stride = 1, pad = (k - 1) * dil / 2, prologue = false, bias = false, residual = false,
                           accumulate = false }, { TX, TY, CO_PT, L_PT }) => {
  if (!Number.isInteger(pad)) throw new Error('convolution padding must be an integer');
  const CO_T = TY * CO_PT, L_T = TX * L_PT, NT = TX * TY, LV = L_PT / 4, CV = CO_PT / 4;
  const tw = stride * L_T + (k - 1) * dil;   // input positions one workgroup reads per channel
  const CI_C = 8 * (tw + k * CO_T) * 4 <= SHARED ? 8 : 4; // input channels per pass
  if (CI_C * 4 * (tw + k * CO_T) > SHARED) throw new Error('convolution tile exceeds workgroup storage');
  const acc = (i, h) => `a${i}_${h}`, range = (n) => [...Array(n).keys()];
  const xAt = (h) => range(4).map((c) => `xs[xb + ${(h * 4 + c) * TX * stride}u]`).join(', ');
  const sum = (v) => accumulate ? `(${v}${bias ? ' + bv' : ''}) + dst[co * p.len + l]`
    : `${v}${bias ? ' + bv' : ''}${residual ? ' + res[co * p.len + l]' : ''}`;
  const store = range(CO_PT).map((i) => `
    { let co = co0 + ty * ${CO_PT}u + ${i}u;
      if (co < p.cout) {${bias ? ' let bv = bias[co];' : ''}
${range(L_PT).map((j) => `        { let l = l0 + tx + ${j * TX}u; if (l < p.len) { dst[co * p.len + l] = ${sum(`${acc(i, j >> 2)}[${j & 3}]`)}; } }`).join('\n')}
      } }`).join('');
  return `
struct P { len: u32, cin: u32, cout: u32, lin: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
${bias ? '@group(0) @binding(4) var<storage, read> bias: array<f32>;' : ''}
${residual ? '@group(0) @binding(5) var<storage, read> res: array<f32>;' : ''}
${prologue ? `@group(0) @binding(6) var<storage, read> pro: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> ia: array<f32>;` : ''}
var<workgroup> xs: array<f32, ${CI_C * tw}>;
var<workgroup> ws: array<vec4<f32>, ${CI_C * k * CO_T / 4}>;
@compute @workgroup_size(${NT})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let tx = tid % ${TX}u; let ty = tid / ${TX}u; let l0 = wg.x * ${L_T}u; let co0 = wg.y * ${CO_T}u;
${range(CO_PT).flatMap((i) => range(LV).map((h) => `  var ${acc(i, h)} = vec4<f32>(0.0);`)).join('\n')}
  for (var ci0 = 0u; ci0 < p.cin; ci0 += ${CI_C}u) {
    for (var i = tid; i < ${CI_C * tw}u; i += ${NT}u) {
      let cl = i / ${tw}u; let pos = i32(${stride}u * l0 + i - cl * ${tw}u) - ${pad};
      var v = 0.0;
      if (pos >= 0 && pos < i32(p.lin)) {
        v = src[(ci0 + cl) * p.lin + u32(pos)];
        ${prologue ? `let q = pro[ci0 + cl]; let t = (v - q.x) * q.y + q.z; let sv = sin(q.w * t); v = t + ia[ci0 + cl] * (sv * sv);` : ''}
      }
      xs[i] = v;
    }
    for (var i = tid; i < ${CI_C * k * CO_T / 4}u; i += ${NT}u) {
      let c4 = i % ${CO_T / 4}u; var v = vec4<f32>(0.0);
      if (co0 + c4 * 4u < p.cout) { v = w[(ci0 * ${k}u + i / ${CO_T / 4}u) * (p.cout / 4u) + co0 / 4u + c4]; }
      ws[i] = v;
    }
    workgroupBarrier();
    for (var cl = 0u; cl < ${CI_C}u; cl++) {
      for (var k = 0u; k < ${k}u; k++) {
        let wb = ((cl * ${k}u + k) * ${CO_T}u + ty * ${CO_PT}u) / 4u;
        let xb = cl * ${tw}u + k * ${dil}u + ${stride}u * tx;
${range(LV).map((h) => `        let x${h} = vec4(${xAt(h)});`).join('\n')}
${range(CV).map((v) => `        let w${v} = ws[wb + ${v}u];`).join('\n')}
${range(CO_PT).flatMap((i) => range(LV).map((h) => `        ${acc(i, h)} = fma(vec4(w${i >> 2}[${i & 3}]), x${h}, ${acc(i, h)});`)).join('\n')}
      }
    }
    workgroupBarrier();
  }
${store}
}`;
};

// Transposed convolution, eight output channels per thread (cout a multiple of 8): out[co][t] = bias[co] + sum over
// taps kk with (t + pad - kk) divisible by stride of w[ci][kk][co] * in[ci][(t + pad - kk) / stride]. The weights are
// packed [ci][k][cout] as vec4s.
export const transposedVecWGSL = ({ k, stride, pad, bias = false }) => {
  const init = (o) => (bias ? `vec4<f32>(bias[o + ${o}u], bias[o + ${o + 1}u], bias[o + ${o + 2}u], bias[o + ${o + 3}u])` : 'vec4<f32>(0.0)');
  return `
struct P { len: u32, cin: u32, cout: u32, lin: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
${bias ? '@group(0) @binding(4) var<storage, read> bias: array<f32>;' : ''}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x; let o = g.y * 8u;
  if (t >= p.len || o >= p.cout) { return; }
  let u = i32(t) + ${pad};
  var a0 = ${init(0)};
  var a1 = ${init(4)};
  for (var kk = u % ${stride}; kk < ${k}; kk += ${stride}) {
    let i = (u - kk) / ${stride};
    if (i >= 0 && i < i32(p.lin)) {
      for (var ci = 0u; ci < p.cin; ci++) {
        let s = vec4<f32>(src[ci * p.lin + u32(i)]);
        let wb = ((ci * ${k}u + u32(kk)) * p.cout + o) / 4u;
        a0 = fma(s, w[wb], a0);
        a1 = fma(s, w[wb + 1u], a1);
      }
    }
  }
  dst[o * p.len + t] = a0.x; dst[(o + 1u) * p.len + t] = a0.y; dst[(o + 2u) * p.len + t] = a0.z; dst[(o + 3u) * p.len + t] = a0.w;
  dst[(o + 4u) * p.len + t] = a1.x; dst[(o + 5u) * p.len + t] = a1.y; dst[(o + 6u) * p.len + t] = a1.z; dst[(o + 7u) * p.len + t] = a1.w;
}`;
};

// Elementwise, one thread per (sample, channel): out[c][t] = op(x[c][j], y[c][t]), with j = t, or with a reflection pad of
// one on the left (x then has lin = len - 1 positions per channel and j is the reflection of t - 1).
export const elementwiseWGSL = (op, reflect = false) => `
struct P { len: u32, lin: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x; let c = g.y;
  if (t >= p.len) { return; }
  ${reflect ? 'var j = i32(t) - 1; if (j < 0) { j = -j; }' : 'let j = i32(t);'}
  let xv = x[c * p.lin + u32(j)];
  let yv = y[c * p.len + t];
  o[c * p.len + t] = ${op};
}`;

// conv_post's 22 rows to the inverse STFT's inputs: rows 0..10 are log magnitudes, rows 11..21 phases before sin. Output
// rows 0..10 are re = exp(a) cos(sin b), rows 11..21 im = exp(a) sin(sin b) (magnitude times (cos ph, sin ph)).
export const polarWGSL = `
struct P { n: u32, len: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  let r = i / p.len; let t = i - r * p.len;
  let k = select(r, r - 11u, r >= 11u);
  let a = exp(x[k * p.len + t]);
  let ph = sin(x[(k + 11u) * p.len + t]);
  o[i] = select(a * cos(ph), a * sin(ph), r >= 11u);
}`;

// Inverse real DFT of one frame per thread (as torch.istft's irfft): frame[f][n] = window[n] / 20 *
// (re0 + (-1)^n re10 + 2 sum_{k=1..9} (re_k cos(2 pi k n / 20) - im_k sin(2 pi k n / 20))). The imaginary parts of the DC
// and Nyquist bins are ignored. The cos and sin basis is a constant buffer [k][n] (bins 0..10, 20 samples each).
// Input: re/im as the polar output [22][len] with rows 0..10 re and 11..21 im. Output: frames [len][20] (frame-major).
export const istftFrameWGSL = `
struct P { len: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;
@group(0) @binding(3) var<storage, read> win: array<f32>;
@group(0) @binding(4) var<storage, read_write> frames: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let f = g.x;
  if (f >= p.len) { return; }
  let bins = 11u;
  var re: array<f32, 11>;
  var im: array<f32, 11>;
  for (var k = 0u; k < bins; k++) { re[k] = x[k * p.len + f]; im[k] = x[(k + 11u) * p.len + f]; }
  for (var n = 0u; n < 20u; n++) {
    var acc = re[0] + select(-re[10], re[10], n % 2u == 0u);
    for (var k = 1u; k < 10u; k++) {
      acc += 2.0 * (re[k] * cs[k * 40u + n] - im[k] * cs[k * 40u + 20u + n]);
    }
    frames[f * 20u + n] = acc * (1.0 / 20.0) * win[n];
  }
}`;

// Overlap-add of the frames and division by the squared-window envelope (torch.istft with center=True).
// Frame f starts at padded position 5 f; output sample i is padded position i + 10 (the first n_fft / 2 are trimmed).
// Output length N = 5 (len - 1). One thread per output sample, gathering the frames that reach it in increasing f:
// f in [ceil((pos - 19) / 5), min(pos / 5, len - 1)], at most four frames.
export const istftOlaWGSL = `
struct P { n: u32, len: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> frames: array<f32>;
@group(0) @binding(2) var<storage, read> win: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  let pos = i + 10u;
  let lo = select(0u, (pos - 15u) / 5u, pos >= 19u);
  let hi = min(pos / 5u, p.len - 1u);
  var s = 0.0; var e = 0.0;
  for (var f = lo; f <= hi; f++) {
    let start = 5u * f;
    if (pos >= start && pos < start + 20u) {
      let n = pos - start;
      s += frames[f * 20u + n];
      e += win[n] * win[n];
    }
  }
  o[i] = s / e;
}`;
