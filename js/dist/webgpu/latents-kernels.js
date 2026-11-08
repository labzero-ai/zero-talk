// WebGPU kernels for the student's acoustic pre-pass. Activations are channel-major [c][t] f32 over the T columns of a pass, except the mixer's recurrence,
// which reads and writes [t][c] (qkvT, gT, yT) so that the values of one step are contiguous. Every kernel takes T (and any element
// count) from its uniform and never from a buffer's size. The Linears are the generator's tiled convolution (kernels.js, k = 1)
// with packed weights (latents.js). The arithmetic follows the reference implementation in its order of operations.
// The acoustic blocks run in chunks of columns (latents.js): the kernels whose result depends on the earlier columns have a chunked form
// (DEPTHWISE_CHUNK, SCAN_CHUNK, RECURRENCE_CHUNK) that starts from the state the last chunk left (the six columns of history, the running
// energy, the delta-rule memory) and does each column's arithmetic in the same order as the whole-sentence form. A chunk's place in the sentence comes
// from a small uniform `cu` = { g0: the chunk's first column in the sentence, f: the sentence's columns, n: the text encoder's tokens, 0 }.

const range = (n) => [...Array(n).keys()];

// Exact GELU: x Phi(x) with Phi from the rational Chebyshev erfc (Numerical Recipes' erfcc).
export const GELU_WGSL = `
fn gelu(x: f32) -> f32 {
  let z = x * 0.7071067811865476;
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.5 * a);
  let poly = -a * a - 1.2655122 + t * (1.0000237 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
  let erfc_a = t * exp(poly);
  return x * select(0.5 * erfc_a, 1.0 - 0.5 * erfc_a, z >= 0.0);
}`;

// Condition: x[c][t] = (bias[c] + w0[c] f0[t] + w1[c] energy[t]) + frames[c][t]. Weights cw are [384][2], row-major.
export const CONDITION = `
struct P { t: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> fr: array<f32>;
@group(0) @binding(2) var<storage, read> f0: array<f32>;
@group(0) @binding(3) var<storage, read> en: array<f32>;
@group(0) @binding(4) var<storage, read> cw: array<f32>;
@group(0) @binding(5) var<storage, read> cb: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  let c = g.y;
  if (t >= p.t) { return; }
  let i = c * p.t + t;
  o[i] = cb[c] + cw[2u * c] * f0[t] + cw[2u * c + 1u] * en[t] + fr[i];
}`;

// The chunked condition: the same sum as CONDITION for the columns g0 .. g0 + T of the sentence, whose frames fr are [384][f] and f0 and energy [f].
export const CONDITION_CHUNK = `
struct P { t: u32 }
struct C { g0: u32, f: u32, n: u32, z: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> fr: array<f32>;
@group(0) @binding(2) var<storage, read> f0: array<f32>;
@group(0) @binding(3) var<storage, read> en: array<f32>;
@group(0) @binding(4) var<storage, read> cw: array<f32>;
@group(0) @binding(5) var<storage, read> cb: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@group(0) @binding(7) var<uniform> cu: C;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  let c = g.y;
  if (t >= p.t) { return; }
  let s = cu.g0 + t;
  o[c * p.t + t] = cb[c] + cw[2u * c] * f0[s] + cw[2u * c + 1u] * en[s] + fr[c * cu.f + s];
}`;

// Depthwise convolution, kernel 7, zero padding: out[c][t] = bias[c] + sum_j w[c][j] x[c][t - 6 + after + j] The
// acoustic pre-pass is causal (after = 0, zero history); the text encoder's blocks see `after` columns past t (its lookahead) and 6 - after before it.
export const DEPTHWISE = `
struct P { t: u32, after: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  let c = g.y;
  if (t >= p.t) { return; }
  var acc = bias[c];
  for (var j = 0u; j < 7u; j++) {
    let s = t + j + p.after;
    if (s >= 6u && s - 6u < p.t) { acc = acc + w[c * 7u + j] * x[c * p.t + s - 6u]; }
  }
  o[c * p.t + t] = acc;
}`;

// The chunked causal depthwise convolution (after = 0): the same sum as DEPTHWISE, in the same order of taps, where the six columns before the
// chunk are in hist [384][6] (the last six columns of the previous chunk's input, saved by HISTORY) and columns before the sentence's first are zero padding (skipped).
export const DEPTHWISE_CHUNK = `
struct P { t: u32 }
struct C { g0: u32, f: u32, n: u32, z: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> hist: array<f32>;
@group(0) @binding(6) var<uniform> cu: C;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  let c = g.y;
  if (t >= p.t) { return; }
  var acc = bias[c];
  for (var j = 0u; j < 7u; j++) {
    let s = t + j;
    if (cu.g0 + s >= 6u) {
      var v = 0.0;
      if (s >= 6u) { v = x[c * p.t + s - 6u]; } else { v = hist[c * 6u + s]; }
      acc = acc + w[c * 7u + j] * v;
    }
  }
  o[c * p.t + t] = acc;
}`;

// The last six columns of a chunk's block input into hist [384][6], for the next chunk's DEPTHWISE_CHUNK (a chunk shorter than six is the sentence's last).
export const HISTORY = `
struct P { t: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= 2304u || p.t < 6u) { return; }
  let c = i / 6u;
  let k = i - c * 6u;
  hist[i] = x[c * p.t + p.t - 6u + k];
}`;

// A chunk's columns [r][t] (stride the chunk's T) into rows of the sentence's buffer [r][f] at the columns g0 .. g0 + T.
export const PLACE = `
struct P { t: u32 }
struct C { g0: u32, f: u32, n: u32, z: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> cu: C;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  let r = g.y;
  if (t >= p.t) { return; }
  dst[r * cu.f + cu.g0 + t] = src[r * p.t + t];
}`;

// LayerNorm over the 384 channels of each column, one thread per column: the channels of a column are read by the threads of a
// workgroup at the same channel, so the loads are coalesced. f32 two-pass mean and variance (population), eps 1e-5.
export const LAYERNORM = `
struct P { t: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bb: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= p.t) { return; }
  var mean = 0.0;
  for (var c = 0u; c < 384u; c++) { mean = mean + x[c * p.t + t]; }
  mean = mean / 384.0;
  var s = 0.0;
  for (var c = 0u; c < 384u; c++) {
    let d = x[c * p.t + t] - mean;
    s = s + d * d;
  }
  let rstd = 1.0 / sqrt(s / 384.0 + 1e-5);
  for (var c = 0u; c < 384u; c++) {
    o[c * p.t + t] = (x[c * p.t + t] - mean) * rstd * w[c] + bb[c];
  }
}`;

// GELU in place over n = 384 T values.
export const GELU = `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
${GELU_WGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  a[i] = gelu(a[i]);
}`;

// Softplus in place over n values: x for x > 20, else ln(1 + e^x). ln(1 + e) is Kahan's log1p, log(u) e / (u - 1) with u = 1 + e,
// which stays accurate where e is below the float32 spacing at 1 (there u = 1 and the result is e, as ln_1p(e) = e to that precision).
export const SOFTPLUS = `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
fn softplus(x: f32) -> f32 {
  if (x > 20.0) { return x; }
  let e = exp(x);
  let u = 1.0 + e;
  if (u == 1.0) { return e; }
  return log(u) * e / (u - 1.0);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  a[i] = softplus(a[i]);
}`;

// Frame expansion: x[c][f] = feats[c][tok(f)] + (pw[c] phase(f) + pb[c]) over the F frames of the expanded sentence. feats is
// [384][n] (stride n: the uniform's n, rewritten before each pass); tok and phase give each frame's token and its phase i / count (float32).
export const EXPAND = `
struct P { f: u32, n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> feats: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read> phase: array<f32>;
@group(0) @binding(4) var<storage, read> pw: array<f32>;
@group(0) @binding(5) var<storage, read> pb: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let f = g.x;
  let c = g.y;
  if (f >= p.f) { return; }
  let v = pw[c] * phase[f] + pb[c];
  o[c * p.f + f] = feats[c * p.n + tok[f]] + v;
}`;

// The frame expansion of the columns g0 .. g0 + T of the sentence (the chunk's own columns [384][T]); tok and phase are the sentence's.
export const EXPAND_CHUNK = `
struct P { t: u32 }
struct C { g0: u32, f: u32, n: u32, z: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> feats: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read> phase: array<f32>;
@group(0) @binding(4) var<storage, read> pw: array<f32>;
@group(0) @binding(5) var<storage, read> pb: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@group(0) @binding(7) var<uniform> cu: C;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let f = g.x;
  let c = g.y;
  if (f >= p.t) { return; }
  let s = cu.g0 + f;
  let v = pw[c] * phase[s] + pb[c];
  o[c * p.t + f] = feats[c * cu.n + tok[s]] + v;
}`;

// The cumulative energy of each hidden channel from a fresh state: cum[c][t] = sum over s <= t of h[c][s]^2. One workgroup per
// channel scans the row in chunks of 256 with an inclusive Hillis-Steele scan in workgroup memory (float32 sums in a different
// order from a sequential loop: rounding only).
export const SCAN = `
struct P { t: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> cum: array<f32>;
var<workgroup> sh: array<f32, 256>;
var<workgroup> carry: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let c = wg.x;
  if (i == 0u) { carry = 0.0; }
  for (var base = 0u; base < p.t; base += 256u) {
    let t = base + i;
    var v = 0.0;
    if (t < p.t) {
      let h = x[c * p.t + t];
      v = h * h;
    }
    sh[i] = v;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off * 2u) {
      var pv = 0.0;
      if (i >= off) { pv = sh[i - off]; }
      workgroupBarrier();
      sh[i] = sh[i] + pv;
      workgroupBarrier();
    }
    if (t < p.t) { cum[c * p.t + t] = carry + sh[i]; }
    workgroupBarrier();
    if (i == 0u) { carry = carry + sh[255]; }
    workgroupBarrier();
  }
}`;

// SCAN for a chunk that starts at a multiple of 256 columns: the running energy enters from st [384] (the last chunk's total; zero at the sentence's
// first column) and the chunk's total is left there. The pieces of 256 and their order are those of the whole row.
export const SCAN_CHUNK = `
struct P { t: u32 }
struct C { g0: u32, f: u32, n: u32, z: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> cum: array<f32>;
@group(0) @binding(3) var<storage, read_write> st: array<f32>;
@group(0) @binding(4) var<uniform> cu: C;
var<workgroup> sh: array<f32, 256>;
var<workgroup> carry: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let c = wg.x;
  if (i == 0u) { carry = select(0.0, st[c], cu.g0 > 0u); }
  workgroupBarrier();
  for (var base = 0u; base < p.t; base += 256u) {
    let t = base + i;
    var v = 0.0;
    if (t < p.t) {
      let h = x[c * p.t + t];
      v = h * h;
    }
    sh[i] = v;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off * 2u) {
      var pv = 0.0;
      if (i >= off) { pv = sh[i - off]; }
      workgroupBarrier();
      sh[i] = sh[i] + pv;
      workgroupBarrier();
    }
    if (t < p.t) { cum[c * p.t + t] = carry + sh[i]; }
    workgroupBarrier();
    if (i == 0u) { carry = carry + sh[255]; }
    workgroupBarrier();
  }
  if (i == 0u) { st[c] = carry; }
}`;

// The response's scale per column: scale[t] = (sum over channels of sqrt(cum[c][t] + 1e-6)) / 384 + 1e-6, the channel sum in order.
export const SCALE = `
struct P { t: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> cum: array<f32>;
@group(0) @binding(2) var<storage, read_write> sc: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= p.t) { return; }
  var s = 0.0;
  for (var c = 0u; c < 384u; c++) { s = s + sqrt(cum[c * p.t + t] + 1e-6); }
  sc[t] = s / 384.0 + 1e-6;
}`;

// h2 = h + gamma h response + beta with response = sqrt(cum + 1e-6) / scale[t]; n = 384 T; the output is a separate buffer.
export const RESPONSE = `
struct P { t: u32, n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> hin: array<f32>;
@group(0) @binding(2) var<storage, read> cum: array<f32>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read> gam: array<f32>;
@group(0) @binding(5) var<storage, read> bet: array<f32>;
@group(0) @binding(6) var<storage, read_write> hout: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  let c = i / p.t;
  let t = i - c * p.t;
  let v = hin[i];
  let response = sqrt(cum[i] + 1e-6) / sc[t];
  hout[i] = v + gam[c] * v * response + bet[c];
}`;

// Tiled transpose: out[t][r] = in[r][t] for in [R][T] (R = the uniform r). Workgroup 32 x 8, a 32 x 33 tile in workgroup memory.
export const TRANSPOSE = `
struct P { t: u32, r: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
var<workgroup> tile: array<f32, 1056>;
@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t0 = wg.x * 32u;
  let r0 = wg.y * 32u;
  for (var i = 0u; i < 4u; i++) {
    let r = r0 + lid.y + 8u * i;
    let t = t0 + lid.x;
    if (r < p.r && t < p.t) { tile[(lid.y + 8u * i) * 33u + lid.x] = src[r * p.t + t]; }
  }
  workgroupBarrier();
  for (var i = 0u; i < 4u; i++) {
    let t = t0 + lid.y + 8u * i;
    let r = r0 + lid.x;
    if (t < p.t && r < p.r) { dst[t * p.r + r] = tile[lid.x * 33u + lid.y + 8u * i]; }
  }
}`;

// LayerNorm over the 64 values of each head (out_norm, shared by the heads) and the SiLU gate: out[c][t] = LN(y)[c] * silu(g[c][t]).
// One thread per (column, head): the head's 64 values are contiguous in yT. Grid (ceil(T / 64), 6).
export const GATE_NORM = `
struct P { t: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> yt: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bb: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let h = gid.y;
  if (t >= p.t) { return; }
  let base = t * 384u + h * 64u;
  var mean = 0.0;
  for (var j = 0u; j < 64u; j++) { mean = mean + yt[base + j]; }
  mean = mean / 64.0;
  var s = 0.0;
  for (var j = 0u; j < 64u; j++) {
    let d = yt[base + j] - mean;
    s = s + d * d;
  }
  let rstd = 1.0 / sqrt(s / 64.0 + 1e-5);
  for (var j = 0u; j < 64u; j++) {
    let c = h * 64u + j;
    let gv = g[c * p.t + t];
    o[c * p.t + t] = ((yt[base + j] - mean) * rstd * w[j] + bb[j]) * (gv * (1.0 / (1.0 + exp(-gv))));
  }
}`;

// The delta-rule mixer's inputs that do not depend on the memory (PREPARE; one thread per (frame, head), grid (ceil(T / 64), 6)): the
// reciprocal of each head's L2 norm of q and of k, 1 / max(norm, 1e-12) with the norm summed in order of i, into nrm
// [t][12] (q in 0..5, k in 6..11); and the gates become alpha = sigmoid(decay row head) and beta = sigmoid(write row 6 + head), in place
// in gT. The serial recurrence then does not repeat the norms or the exponentials per step.
export const PREPARE = `
struct P { t: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> gt: array<f32>;
@group(0) @binding(3) var<storage, read_write> nrm: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  let head = g.y;
  if (t >= p.t) { return; }
  let q = t * 1152u + head * 64u;
  let k = q + 384u;
  var qs = 0.0;
  var ks = 0.0;
  for (var i = 0u; i < 64u; i++) {
    let a = qkv[q + i];
    let c = qkv[k + i];
    qs = qs + a * a;
    ks = ks + c * c;
  }
  let gi = t * 12u + head;
  nrm[gi] = 1.0 / max(sqrt(qs), 1e-12);
  nrm[gi + 6u] = 1.0 / max(sqrt(ks), 1e-12);
  gt[gi] = 1.0 / (1.0 + exp(-gt[gi]));
  gt[gi + 6u] = 1.0 / (1.0 + exp(-gt[gi + 6u]));
}`;

// The delta-rule mixer's recurrence one workgroup per head (6 workgroups of 64 threads); thread j owns
// column j of the head's 64 x 64 memory in 64 registers. Per step: the norms' reciprocals and the gates come from PREPARE (nrm, gT), and
// the raw q and k are in workgroup memory: M = alpha M; acc_j = sum_i (k_i / |k|) M[i][j]; r_j = v_j - acc_j; M[i][j] += beta (k_i/|k|) r_j;
// y_j = sum_i (q_i / |q|) M[i][j], the sums in order of i. The write loop reads its own copy of k (sk2), so its products are not shared
// with the decay loop. Columns are independent, so the only sharing is of q and k, through workgroup memory
// (double-buffered by the step's parity: one barrier per step).
// Inputs: qkvT [t][1152] (q, k, v blocks of 384 each, head-major within a block), gT [t][12] (alpha and beta), nrm [t][12]; output yT [t][384].
// The loops over the 64 rows are unrolled.
// The chunked form (RECURRENCE_CHUNK) starts a chunk from the memory the last one left in mem [6][64][64] (thread j's column of each row, at
// (head * 64 + i) * 64 + j; zero at the sentence's first column) and stores it back at the end; the steps are the same.
const recurrence = (chunked) => {
  const decl = range(64).map((i) => `  var m${i}: f32 = 0.0;`).join('\n');
  const load = range(64).map((i) => `    m${i} = mem[mb + ${i * 64}u];`).join('\n');
  const store = range(64).map((i) => `  mem[mb + ${i * 64}u] = m${i};`).join('\n');
  const decay = range(64).map((i) => `    m${i} = m${i} * al; acc = acc + (sk[b + ${i}u] * kf) * m${i};`).join('\n');
  const write = range(64).map((i) => `    let wk${i} = be * (sk2[b + ${i}u] * kf); m${i} = m${i} + wk${i} * rr; acc2 = acc2 + (sq[b + ${i}u] * qf) * m${i};`).join('\n');
  return `
struct P { t: u32 }
${chunked ? 'struct C { g0: u32, f: u32, n: u32, z: u32 }' : ''}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> gt: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> nrm: array<f32>;
${chunked ? `@group(0) @binding(5) var<storage, read_write> mem: array<f32>;
@group(0) @binding(6) var<uniform> cu: C;` : ''}
var<workgroup> sq: array<f32, 128>;
var<workgroup> sk: array<f32, 128>;
var<workgroup> sk2: array<f32, 128>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let head = wg.x;
  let hj = head * 64u + j;
${decl}
${chunked ? `  let mb = head * 4096u + j;
  if (cu.g0 > 0u) {
${load}
  }` : ''}
  var nq = qkv[hj];
  var nk = qkv[384u + hj];
  var nv = qkv[768u + hj];
  var na = gt[head];
  var nb = gt[6u + head];
  var nqf = nrm[head];
  var nkf = nrm[6u + head];
  for (var t = 0u; t < p.t; t++) {
    let b = (t & 1u) * 64u;
    let vv = nv;
    let al = na;
    let be = nb;
    let qf = nqf;
    let kf = nkf;
    sq[b + j] = nq;
    sk[b + j] = nk;
    sk2[b + j] = nk;
    if (t + 1u < p.t) {
      let base = (t + 1u) * 1152u;
      nq = qkv[base + hj];
      nk = qkv[base + 384u + hj];
      nv = qkv[base + 768u + hj];
      na = gt[(t + 1u) * 12u + head];
      nb = gt[(t + 1u) * 12u + 6u + head];
      nqf = nrm[(t + 1u) * 12u + head];
      nkf = nrm[(t + 1u) * 12u + 6u + head];
    }
    workgroupBarrier();
    var acc = 0.0;
${decay}
    let rr = vv - acc;
    var acc2 = 0.0;
${write}
    y[t * 384u + hj] = acc2;
  }
${chunked ? store : ''}
}`;
};
export const RECURRENCE = recurrence(false);
export const RECURRENCE_CHUNK = recurrence(true);
