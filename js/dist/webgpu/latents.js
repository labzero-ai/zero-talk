// The student's acoustic pre-pass on WebGPU : one push of a whole sentence from a fresh state, with the latents [512][T] left on the GPU. The weights come from the student part of the model, checked
// against the settings and shapes the kernels assume. The kernels are in latents-kernels.js and the Linears are the generator's tiled
// convolution (kernels.js, k = 1). Built like generator.js: an arena sized for the current capacity and shared by every length,
// a plan of calls per length with its uniforms and bind groups, and one readback buffer for the latents. A run
// is one compute pass, in call order.
// The text encoder: textInput gives x0 [384][n]; encode runs the text blocks (the acoustic ConvNeXt block, with the depthwise
// context centred on the lookahead) and leaves the features [384][n] in FEATS, and reads back the durations (40 Hz units, before the
// rounding). The frame pass: frames(counts40, F) expands the features to F frames (XA), runs the frame block into XB, which is
// where run(null, ...) takes the frames, and reads back the raw F0 (after the softplus) and energy.
import { convWGSL, TILE } from './kernels.js';
import * as KN from './latents-kernels.js';

// The buffer usage flags, as the WebGPU spec fixes them, so this module also loads where WebGPU is missing (Node.js, some
// browsers); it is only used where it exists.
const GPUBufferUsage = globalThis.GPUBufferUsage ?? { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 };
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const READBACK = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

export const WIDTH = 384;          // the stream, the hidden layer and the mixers' width
export const FEATURES = 3 * WIDTH; // the mixers' q, k and v: the largest buffer, 1152 words per frame
export const DECAY = 12;           // the mixers' decay (6 heads) and write (6 heads) gates
export const HEADS = 6;
export const HEAD_DIM = 64;
export const KERNEL = 7;
export const LATENTS = 512;
export const BLOCKS = 6;
export const MIXER_AFTER = [1, 4];
export const MAX_FRAMES = 2600;    // the most frames of a sentence the acoustic pre-pass takes; 504 tokens already reach 2,262 frames
// The arena's capacities in frames: a growth goes to the first of these that holds the sentence (never beyond MAX_FRAMES).
export const BUCKETS = [600, 1024, 1536, 2048, MAX_FRAMES];
export const MAX_TOKENS = 512;     // the most tokens of a sentence the text encoder takes; the smallest arena holds 600 frames
export const COORDS = 4;           // the voice's length coordinates (voice_fold has 1 + COORDS rows of WIDTH)
// The arena's buffers in words per frame (a run of T frames uses the first T frames of each row). The frames share the storage of XB
// (the condition reads them before block 0's down projection writes XB), and the mixer's HN, GP, YT and GM share those of the block
// buffers N, H, D and H2, which are free after the last block (each kernel reads its inputs before its output is written, see buildPlan).
// The mixer's reciprocal norms [t][12] (PREPARE) share CUM: a mixer does not otherwise use it, and the next block's energy scan writes it after the mixer.
export const PER_FRAME = { XA: WIDTH, XB: WIDTH, D: WIDTH, N: WIDTH, H: WIDTH, H2: WIDTH, CUM: WIDTH, SC: 1, F0: 1, EN: 1,
  QKV: FEATURES, QKVT: FEATURES, G: DECAY, GT: DECAY };
// The latents are one [512][T] buffer per arena: a run's latents are valid until the next run is submitted (the worker reads a sentence's
// windows before it prepares the next sentence, so no window reads an overwritten run).
export const WORDS_PER_FRAME = Object.values(PER_FRAME).reduce((a, b) => a + b, 0) + LATENTS;
// The bytes of an arena for `frames` frames (its activations and latents) and of its readback buffer.
export const arenaBytes = (frames) => 4 * WORDS_PER_FRAME * frames;
export const readbackBytes = (frames) => 4 * LATENTS * frames;
const MAX_PLANS = 16;
const CONV_L = TILE.TX * TILE.L_PT;   // columns per convolution workgroup
const CONV_CO = TILE.TY * TILE.CO_PT; // output channels per convolution workgroup
const range = (n) => [...Array(n).keys()];

// PyTorch's Linear weight [cout][cin], packed for convWGSL with k = 1 as [cin][cout] (cout is a multiple of 4).
const packLinear = (w, cout, cin) => {
  const out = new Float32Array(cin * cout);
  for (let co = 0; co < cout; co++) for (let ci = 0; ci < cin; ci++) out[ci * cout + co] = w[co * cin + ci];
  return out;
};

// The model must be the student this step implements: its settings (expansion 1, student_acoustic_depth 6, and
// heads * head_dim = width), the text encoder's lookahead (a whole number of columns per text block, at most KERNEL - 1) and the shapes and
// mixer placement the kernels assume. Throws with the entry that differs.
export const checkModel = (part) => {
  const meta = (k) => {
    const v = part.meta[`student_${k}`];
    if (v === undefined) throw new Error(`acoustic: the model has no student_${k} entry`);
    return Number(v);
  };
  const width = meta('width'), kernel = meta('kernel'), heads = meta('heads'), headDim = meta('head_dim');
  const depth = meta('acoustic_depth'), expansion = meta('expansion');
  const textDepth = meta('text_depth'), lookahead = meta('text_lookahead');
  const want = (what, got, expected) => { if (got !== expected) throw new Error(`acoustic: the model's ${what} is ${got}, this step needs ${expected}`); };
  want('student_width', width, WIDTH);
  want('student_kernel', kernel, KERNEL);
  want('student_heads', heads, HEADS);
  want('student_head_dim', headDim, HEAD_DIM);
  want('student_heads * student_head_dim', heads * headDim, width);
  want('student_acoustic_depth', depth, BLOCKS);
  want('student_expansion', expansion, 1);
  if (!(textDepth >= 1) || lookahead % textDepth !== 0 || lookahead / textDepth > KERNEL - 1) {
    throw new Error(`acoustic: the model's text_lookahead ${lookahead} must be a multiple of text_depth ${textDepth}, at most ${KERNEL - 1} columns per block`);
  }
  const shape = (name, expected) => {
    if (!part.has(name)) throw new Error(`acoustic: the model has no tensor ${name}`);
    const s = part.shape(name);
    if (s.join(',') !== expected.join(',')) throw new Error(`acoustic: ${name} has shape [${s}], this step needs [${expected}]`);
  };
  for (let i = 0; i < BLOCKS; i++) shape(`acoustic.blocks.${i}.depthwise.weight`, [WIDTH, 1, KERNEL]);
  for (let i = 0; i < textDepth; i++) shape(`prosody.blocks.${i}.depthwise.weight`, [WIDTH, 1, KERNEL]);
  shape('prosody.duration.weight', [1, WIDTH]);
  shape('prosody.duration.bias', [1]);
  shape('prosody.voice_fold', [1 + COORDS, WIDTH]);
  if (!part.has('prosody.embedding.weight') || part.shape('prosody.embedding.weight')[1] !== WIDTH) {
    throw new Error('acoustic: prosody.embedding.weight must be [vocab][384]');
  }
  shape('prosody.phase.weight', [WIDTH, 1]);
  shape('prosody.phase.bias', [WIDTH]);
  shape('prosody.frame_block.depthwise.weight', [WIDTH, 1, KERNEL]);
  shape('prosody.prosody.weight', [2, WIDTH]);
  shape('prosody.prosody.bias', [2]);
  shape('acoustic.condition.weight', [WIDTH, 2]);
  for (let i = 0; i < BLOCKS; i++) {
    const has = part.has(`acoustic.mixers.${i}.qkv.weight`);
    if (has !== MIXER_AFTER.includes(i)) throw new Error(`acoustic: mixers are needed exactly at indices ${MIXER_AFTER}; acoustic.mixers.${i} is ${has ? 'present' : 'absent'}`);
  }
};

// The most frames the acoustic pre-pass takes on this device: the QKV buffer (1152 words per frame) must fit a storage binding and a buffer, and the
// elementwise dispatches (ceil(384 T / 256) workgroups) must fit the per-dimension workgroup limit.
export const frameLimit = (device) => {
  const l = device.limits;
  return Math.min(Math.floor(l.maxStorageBufferBindingSize / (4 * FEATURES)), Math.floor(l.maxBufferSize / (4 * FEATURES)),
    Math.floor((l.maxComputeWorkgroupsPerDimension * 256) / WIDTH));
};

// The latent statistics of a read-back latent array [512][T] (stride T): per channel the mean over the T frames,
// then ln(population sd + 1e-4), in float64. The result is 1024 float32 values: the 512 means, then the 512 log standard deviations (the
// layout of zt_gpu_moments).
export const latentStatistics = (lat, T) => {
  const out = new Float64Array(1024);
  for (let c = 0; c < 512; c++) {
    let s = 0;
    for (let t = 0; t < T; t++) s += lat[c * T + t];
    const m = s / T;
    let v = 0;
    for (let t = 0; t < T; t++) { const d = lat[c * T + t] - m; v += d * d; }
    out[c] = Math.fround(m);
    out[512 + c] = Math.fround(Math.log(Math.sqrt(v / T) + 1e-4));
  }
  return out;
};

// The source columns of frames [s, e) of a sentence of F frames: har [22][60 F + 1] cut to [22][60 (e - s) + 1], as windows.js's
// windowOf cuts them (the worker's source half of a window; its latents come from the acoustic buffer).
export const sourceWindow = (har, F, s, e) => {
  const LH = 60 * F + 1, Ln = 60 * (e - s) + 1, out = new Float32Array(22 * Ln);
  for (let r = 0; r < 22; r++) out.set(har.subarray(r * LH + 60 * s, r * LH + 60 * e + 1), r * Ln);
  return out;
};

// The text encoder's input x0 [384][n]: x[c][t] = embed[id_t][c] + s[c], where s is the voice fold combined with the coordinates
// in float64 (fold[i] + sum_r coords[r] fold[(1 + r) 384 + i]), rounded to float32, and the sum is float32. tables = { embed, fold }: the
// part's prosody.embedding.weight and prosody.voice_fold as Float32Arrays (GpuAcoustic.tables). A pure function: it keeps nothing.
export const textInput = ({ embed, fold }, ids, coords) => {
  const n = ids.length, vocab = embed.length / WIDTH;
  const s = new Float32Array(WIDTH);
  for (let i = 0; i < WIDTH; i++) {
    let acc = 0;
    for (let r = 0; r < COORDS; r++) acc += coords[r] * fold[(1 + r) * WIDTH + i];
    s[i] = fold[i] + acc;   // stored as float32: the rounding of the host's `as f32`
  }
  const x = new Float32Array(WIDTH * n);
  for (let t = 0; t < n; t++) {
    if (ids[t] >= vocab) throw new Error(`token ${ids[t]} is outside the vocabulary of ${vocab}`);
    for (let c = 0; c < WIDTH; c++) x[c * n + t] = embed[ids[t] * WIDTH + c] + s[c];
  }
  return x;
};

export class GpuAcoustic {
  // An acoustic pre-pass for sentences of up to maxFrames frames (its arena's capacity; ensureCapacity grows it).
  static async create(device, part, { maxFrames = MAX_FRAMES } = {}) {
    const b = new GpuAcoustic(device, part, maxFrames);
    await b.ready;
    await device.queue.onSubmittedWorkDone();
    return b;
  }

  constructor(device, part, maxFrames) {
    checkModel(part);
    if (!Number.isInteger(maxFrames) || maxFrames < 1) throw new Error(`maxFrames must be a positive integer, not ${maxFrames}`);
    if (maxFrames > MAX_FRAMES) throw new Error(`maxFrames ${maxFrames} is above MAX_FRAMES (${MAX_FRAMES}): the frame pass's readbacks are sized for it`);
    const limit = frameLimit(device);
    if (maxFrames > limit) throw new Error(`maxFrames ${maxFrames} is above this device's limit of ${limit} frames (the QKV buffer of 1152 words per frame and the workgroup limit)`);
    Object.assign(this, { device, part, maxFrames, keep: [], arenaBufs: [], plans: new Map(), kerns: new Map(), weightBytes: 0,
      readPending: false, failedFrames: null, textPending: false, textFor: null, framesPending: false, framesFor: null });
    // The recurrence and its prepare dispatch (PREPARE, before each mixer's recurrence).
    const rec = KN.RECURRENCE, prep = KN.PREPARE;
    this.k = {
      conv: this.kern(convWGSL({ k: 1, dil: 1, pad: 0, bias: true }, TILE), 'linear'),
      convRes: this.kern(convWGSL({ k: 1, dil: 1, pad: 0, bias: true, residual: true }, TILE), 'linear + residual'),
      cond: this.kern(KN.CONDITION, 'condition'), dw: this.kern(KN.DEPTHWISE, 'depthwise'), ln: this.kern(KN.LAYERNORM, 'layer norm'),
      gelu: this.kern(KN.GELU, 'gelu'), scan: this.kern(KN.SCAN, 'energy scan'), scale: this.kern(KN.SCALE, 'response scale'),
      resp: this.kern(KN.RESPONSE, 'response'), tr: this.kern(KN.TRANSPOSE, 'transpose'), rec: this.kern(rec, 'delta rule'),
      prep: this.kern(prep, 'delta prep'),
      gnorm: this.kern(KN.GATE_NORM, 'head norm and gate'), sp: this.kern(KN.SOFTPLUS, 'softplus'),
      expand: this.kern(KN.EXPAND, 'frame expansion'),
    };
    this.w = this.loadWeights();
    // The text encoder's features [384][n] (kept, not in the arena: the frame pass reads them) and its durations (4 rows of n; row 0 is
    // read back through dursRead). The raw F0 and energy of frames() come back through framesRead (rows of F, 2 * MAX_FRAMES words).
    this.feats = this.mk(4 * WIDTH * MAX_TOKENS, STORAGE);
    this.durs = this.mk(4 * 4 * MAX_TOKENS, STORAGE);
    this.dursRead = this.mk(4 * MAX_TOKENS, READBACK);
    this.framesRead = this.mk(4 * 2 * MAX_FRAMES, READBACK);
    this.allocArena();
    this.ready = Promise.all([...this.kerns.values()].map(async (k) => {
      k.pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: k.code }), entryPoint: 'main' } });
    }));
  }

  kern(code, label) {
    if (!this.kerns.has(code)) this.kerns.set(code, { code, label, pipe: null });
    return this.kerns.get(code);
  }

  mk(bytes, usage, list = this.keep) {
    const b = this.device.createBuffer({ size: Math.max(16, bytes), usage });
    list.push(b);
    return b;
  }

  upload(data) {
    const b = this.mk(data.byteLength, STORAGE);
    this.device.queue.writeBuffer(b, 0, data);
    this.weightBytes += data.byteLength;
    return b;
  }

  // The student's tensors: every Linear packed for convWGSL; the residual variant for the Linears whose output adds the stream.
  loadWeights() {
    const P = this.part, up = (name) => this.upload(P.f32(name));
    const lin = (name, cout, cin, residual = false) => {
      const s = P.shape(`${name}.weight`);
      if (s[0] !== cout || s[1] !== cin) throw new Error(`${name}: shape [${s}], expected [${cout}, ${cin}]`);
      return { kern: residual ? this.k.convRes : this.k.conv, w: this.upload(packLinear(P.f32(`${name}.weight`), cout, cin)),
        bias: up(`${name}.bias`), cin, cout };
    };
    // A ConvNeXt block under the prefix p: the acoustic blocks, the text encoder's and the frame block.
    const block = (p) => ({ dw: up(`${p}.depthwise.weight`), dwB: up(`${p}.depthwise.bias`), lnW: up(`${p}.norm.weight`), lnB: up(`${p}.norm.bias`),
      up: lin(`${p}.up`, WIDTH, WIDTH), down: lin(`${p}.down`, WIDTH, WIDTH, true), gamma: up(`${p}.gamma`), beta: up(`${p}.beta`) });
    const blocks = range(BLOCKS).map((i) => block(`acoustic.blocks.${i}`));
    const mixers = {};
    for (const i of MIXER_AFTER) {
      const p = `acoustic.mixers.${i}`;
      mixers[i] = { lnW: up(`${p}.norm.weight`), lnB: up(`${p}.norm.bias`), qkv: lin(`${p}.qkv`, FEATURES, WIDTH),
        dec: lin(`${p}.decay_write`, DECAY, WIDTH), gate: lin(`${p}.gate`, WIDTH, WIDTH), outNW: up(`${p}.out_norm.weight`),
        outNB: up(`${p}.out_norm.bias`), out: lin(`${p}.out`, WIDTH, WIDTH, true) };
    }
    const cond = { cw: up('acoustic.condition.weight'), cb: up('acoustic.condition.bias') };
    // The text encoder: its blocks (their depthwise sees `after` columns past each token, see DEPTHWISE) and the duration head, whose one output
    // is padded to four rows as the Linears' tiling needs (rows 1 to 3 are zero and are not read back).
    const textDepth = Number(P.meta.student_text_depth), dw = new Float32Array(4 * WIDTH), db = new Float32Array(4);
    dw.set(P.f32('prosody.duration.weight'));
    db.set(P.f32('prosody.duration.bias'));
    const text = { after: Number(P.meta.student_text_lookahead) / textDepth, blocks: range(textDepth).map((i) => block(`prosody.blocks.${i}`)),
      dur: { kern: this.k.conv, w: this.upload(packLinear(dw, 4, WIDTH)), bias: this.upload(db), cin: WIDTH, cout: 4 } };
    // The frame pass: the phase Linear's per-channel weight and bias, the frame block (an acoustic block with no history, so
    // after = 0) and the prosody Linear [2][384] padded to four rows (row 0: F0 before the softplus, row 1: the energy).
    const pro = new Float32Array(4 * WIDTH), proB = new Float32Array(4);
    pro.set(P.f32('prosody.prosody.weight'));
    proB.set(P.f32('prosody.prosody.bias'));
    const frames = { pw: up('prosody.phase.weight'), pb: up('prosody.phase.bias'), block: block('prosody.frame_block'),
      prosody: { kern: this.k.conv, w: this.upload(packLinear(pro, 4, WIDTH)), bias: this.upload(proB), cin: WIDTH, cout: 4 } };
    this.tables = { embed: P.f32('prosody.embedding.weight'), fold: P.f32('prosody.voice_fold') };
    return { blocks, mixers, cond, out: lin('acoustic.output', LATENTS, WIDTH), text, frames };
  }

  // The activations, the latents and the readback, sized for the current capacity (maxFrames), with the aliases of PER_FRAME.
  // Every buffer goes to arenaBufs, so a growth or destroy frees the whole arena.
  allocArena() {
    const M = this.maxFrames, A = (words) => this.mk(4 * words * M, STORAGE, this.arenaBufs);
    const b = Object.fromEntries(Object.entries(PER_FRAME).map(([name, words]) => [name, A(words)]));
    this.buf = { ...b, FR: b.XB, HN: b.N, GP: b.H, YT: b.D, GM: b.H2, LAT: A(LATENTS) };
    // The one readback buffer: a run made with read copies its latents here, and read() maps it. Runs are read one at a time.
    this.readback = this.mk(readbackBytes(M), READBACK, this.arenaBufs);
  }

  // The capacity for a sentence of T frames: the first bucket that holds T, at most MAX_FRAMES and this device's limit.
  capacityFor(T) {
    const want = BUCKETS.find((b) => b >= T) ?? MAX_FRAMES;
    return Math.min(want, MAX_FRAMES, frameLimit(this.device));
  }

  // Destroys the arena and the cached plans, then makes the arena for M frames in error scopes. The old arena goes first: its memory is
  // needed for the new one. On failure no arena is left (the caller makes one again); the error has oom set when the device ran out of
  // memory.
  async remake(M) {
    for (const p of this.plans.values()) p.uniforms.forEach((u) => u.destroy());
    this.plans.clear();
    this.arenaBufs.forEach((b) => b.destroy());
    this.arenaBufs = [];
    this.maxFrames = M;
    this.framesFor = null;   // frames() left them in XB, which is one of the arena buffers destroyed here
    const d = this.device;
    d.pushErrorScope('validation');
    d.pushErrorScope('out-of-memory');
    let failure = null, oom = null, invalid = null;
    try {
      this.allocArena();
    } catch (e) {
      failure = e;
    } finally {
      oom = await d.popErrorScope();
      invalid = await d.popErrorScope();
    }
    if (failure || oom || invalid) {
      this.arenaBufs.forEach((b) => b.destroy());
      this.arenaBufs = [];
      if (failure) throw failure;
      throw Object.assign(new Error(`the acoustic arena for ${M} frames was not made (${oom ? `out of memory: ${oom.message}` : `validation: ${invalid.message}`})`),
        { oom: oom !== null });
    }
  }

  // Grows the arena to the bucket that holds a sentence of T frames (capacityFor), when T is more than maxFrames. The old arena goes first
  // (remake). When the new one cannot be made, the old size is made again, so the acoustic pre-pass still works at the old size, and the growth throws
  // (oom: out of memory). A size that ran out of memory is remembered: a later sentence that needs it throws at once, without an allocation.
  // When even the old size cannot be made, the error is fatal (fatal = true): the acoustic pre-pass is gone and the caller stops the GPU engine.
  // Needs no run in flight (read() has been called for the last run made with read).
  async ensureCapacity(T) {
    if (!Number.isInteger(T) || T < 1) throw new Error(`T must be a positive integer, not ${T}`);
    if (T <= this.maxFrames) return;
    const want = this.capacityFor(T);
    if (T > want) throw new Error(`a sentence of ${T} frames is above the acoustic pre-pass's limit of ${want} frames`);
    if (this.failedFrames !== null && want >= this.failedFrames) {
      throw new Error(`the acoustic arena for ${want} frames ran out of memory before; a sentence of ${T} frames runs in WebAssembly`);
    }
    if (this.readPending) throw new Error('read the run in flight before the acoustic pre-pass grows');
    const old = this.maxFrames;
    try {
      await this.remake(want);
    } catch (e) {
      if (e.oom) this.failedFrames = this.failedFrames === null ? want : Math.min(this.failedFrames, want);
      try {
        await this.remake(old);
      } catch (e2) {
        throw Object.assign(new Error(`the acoustic arena could not be made again at ${old} frames after a failed growth to ${want} (${e2.message})`),
          { fatal: true });
      }
      throw e;
    }
  }

  // The calls of a plan (cached per key, least recently used first out): a run of T frames is keyed by T, a text pass of n tokens by `text:${n}`,
  // the frame pass of F frames by `frames:${F}`.
  cached(key, build) {
    if (this.plans.has(key)) {
      const p = this.plans.get(key);
      this.plans.delete(key);
      this.plans.set(key, p);
      return p;
    }
    while (this.plans.size >= MAX_PLANS) {
      const [old, p] = this.plans.entries().next().value;
      p.uniforms.forEach((u) => u.destroy());
      this.plans.delete(old);
    }
    const p = build();
    this.plans.set(key, p);
    return p;
  }

  plan(T) {
    if (!Number.isInteger(T) || T < 1 || T > this.maxFrames) throw new Error(`T = ${T} is outside 1..${this.maxFrames}`);
    return this.cached(T, () => this.buildPlan(T));
  }

  textPlan(n) {
    const most = Math.min(MAX_TOKENS, this.maxFrames);
    if (!Number.isInteger(n) || n < 1 || n > most) throw new Error(`n = ${n} tokens is outside 1..${most}`);
    return this.cached(`text:${n}`, () => this.buildTextPlan(n));
  }

  framesPlan(F) {
    if (!Number.isInteger(F) || F < 1 || F > this.maxFrames) throw new Error(`frames: F = ${F} is outside 1..${this.maxFrames}`);
    return this.cached(`frames:${F}`, () => this.buildFramesPlan(F));
  }

  // The builder of a plan of T columns: p.calls (each with its uniforms and bind groups) and the steps they are made of: a Linear, a ConvNeXt
  // block (the acoustic blocks, the text encoder's and the frame block: the same dispatches) and a mixer.
  planner(T) {
    const B = this.buf, k = this.k, p = { T, calls: [], uniforms: [] };
    const uni = (...v) => {
      const u = this.device.createBuffer({ size: 16, usage: UNIFORM });
      this.device.queue.writeBuffer(u, 0, new Uint32Array([...v, 0, 0, 0].slice(0, 4)));
      p.uniforms.push(u);
      return u;
    };
    const disp = (kern, binds, gx, gy, label) => {
      const entries = Object.entries(binds).filter(([, b]) => b).map(([i, b]) => ({ binding: Number(i), resource: { buffer: b } }));
      p.calls.push({ pipe: kern.pipe, bg: this.device.createBindGroup({ layout: kern.pipe.getBindGroupLayout(0), entries }), gx, gy, label });
    };
    // A Linear: out[co][t] = bias[co] + sum_ci W[co][ci] in[ci][t] (+ res[co][t]).
    const linear = (l, src, dst, res, label) => disp(l.kern, { 0: uni(T, l.cin, l.cout, T), 1: src, 2: l.w, 3: dst, 4: l.bias, 5: res },
      Math.ceil(T / CONV_L), Math.ceil(l.cout / CONV_CO), label);
    // A ConvNeXt block from cur into dst = down(...) + cur. after: the depthwise's columns past each t (0 for the causal acoustic block).
    const block = (blk, cur, dst, after, name) => {
      const W = WIDTH * T;
      disp(k.dw, { 0: uni(T, after), 1: cur, 2: blk.dw, 3: blk.dwB, 4: B.D }, Math.ceil(T / 64), WIDTH, `${name} depthwise`);
      disp(k.ln, { 0: uni(T), 1: B.D, 2: blk.lnW, 3: blk.lnB, 4: B.N }, Math.ceil(T / 64), 1, `${name} norm`);
      linear(blk.up, B.N, B.H, null, `${name} up`);
      disp(k.gelu, { 0: uni(W), 1: B.H }, Math.ceil(W / 256), 1, `${name} gelu`);
      disp(k.scan, { 0: uni(T), 1: B.H, 2: B.CUM }, WIDTH, 1, `${name} energy`);
      disp(k.scale, { 0: uni(T), 1: B.CUM, 2: B.SC }, Math.ceil(T / 64), 1, `${name} scale`);
      disp(k.resp, { 0: uni(T, W), 1: B.H, 2: B.CUM, 3: B.SC, 4: blk.gamma, 5: blk.beta, 6: B.H2 }, Math.ceil(W / 256), 1, `${name} response`);
      linear(blk.down, B.H2, dst, cur, `${name} down + residual`);
    };
    // The mixer: its inputs are read by earlier dispatches than the buffers they share with (HN = N is written after the block's up
    // projection has read N; GP = H, YT = D and GM = H2 are free; GM is read only by the output Linear, after GP and YT are read).
    const mixer = (i, X, OUT) => {
      const m = this.w.mixers[i], name = `mixer ${i}`;
      disp(k.ln, { 0: uni(T), 1: X, 2: m.lnW, 3: m.lnB, 4: B.HN }, Math.ceil(T / 64), 1, `${name} norm`);
      linear(m.qkv, B.HN, B.QKV, null, `${name} qkv`);
      linear(m.dec, B.HN, B.G, null, `${name} decay and write`);
      linear(m.gate, B.HN, B.GP, null, `${name} gate`);
      disp(k.tr, { 0: uni(T, FEATURES), 1: B.QKV, 2: B.QKVT }, Math.ceil(T / 32), Math.ceil(FEATURES / 32), `${name} transpose qkv`);
      disp(k.tr, { 0: uni(T, DECAY), 1: B.G, 2: B.GT }, Math.ceil(T / 32), Math.ceil(DECAY / 32), `${name} transpose gates`);
      disp(k.prep, { 0: uni(T), 1: B.QKVT, 2: B.GT, 3: B.CUM }, Math.ceil(T / 64), HEADS, `${name} delta prep`);
      disp(k.rec, { 0: uni(T), 1: B.QKVT, 2: B.GT, 3: B.YT, 4: B.CUM }, HEADS, 1, `${name} delta rule`);
      disp(k.gnorm, { 0: uni(T), 1: B.YT, 2: B.GP, 3: m.outNW, 4: m.outNB, 5: B.GM }, Math.ceil(T / 64), HEADS, `${name} head norm and gate`);
      linear(m.out, B.GM, OUT, X, `${name} out + residual`);
    };
    return { p, disp, uni, linear, block, mixer };
  }

  buildPlan(T) {
    const { p, disp, uni, linear, block, mixer } = this.planner(T);
    const B = this.buf;
    // The condition: x = (bias + w0 f0 + w1 energy) + frames; then the blocks, with the mixers after blocks 1 and 4.
    disp(this.k.cond, { 0: uni(T), 1: B.FR, 2: B.F0, 3: B.EN, 4: this.w.cond.cw, 5: this.w.cond.cb, 6: B.XA }, Math.ceil(T / 64), WIDTH, 'condition');
    let cur = B.XA;
    const other = (x) => (x === B.XA ? B.XB : B.XA);
    for (let i = 0; i < BLOCKS; i++) {
      const nxt = other(cur);
      block(this.w.blocks[i], cur, nxt, 0, `block ${i}`);
      cur = nxt;
      if (MIXER_AFTER.includes(i)) {
        const out = other(cur);
        mixer(i, cur, out);
        cur = out;
      }
    }
    linear(this.w.out, cur, B.LAT, null, 'output');
    return p;
  }

  // The text encoder's plan for n tokens: the D text blocks from XA (x0, written by encode) through XB and XA, the last one into FEATS; then the
  // duration head (its row 0 into durs) and the softplus in place on the n durations.
  buildTextPlan(n) {
    const { p, disp, uni, linear, block } = this.planner(n);
    const B = this.buf, text = this.w.text;
    let cur = B.XA;
    text.blocks.forEach((blk, i) => {
      const dst = i === text.blocks.length - 1 ? this.feats : cur === B.XA ? B.XB : B.XA;
      block(blk, cur, dst, text.after, `text ${i}`);
      cur = dst;
    });
    linear(text.dur, this.feats, this.durs, null, 'duration');
    disp(this.k.sp, { 0: uni(n), 1: this.durs }, Math.ceil(n / 256), 1, 'softplus');
    return p;
  }

  // The frame pass's plan for F frames: the expansion (XA), the frame block (XA -> XB; the frames are XB, where run(null, ...) reads them), the
  // prosody Linear (XB -> QKV, rows 0 and 1 of F) and the softplus of row 0 in place. Aliases, each safe because the pass reads an input before
  // anything writes it: the expansion's token index and phase are read from F0 and EN, which run() fills only after this pass (frames() writes
  // them just before); the prosody output goes to QKV, which the mixers write only inside run(), after the frames have been taken. The
  // expansion's uniform holds n (the features' stride), rewritten by frames() before each pass; the plan itself depends on F alone.
  buildFramesPlan(F) {
    const { p, disp, uni, linear, block } = this.planner(F);
    const B = this.buf, fr = this.w.frames;
    p.nUni = uni(F, 0);
    disp(this.k.expand, { 0: p.nUni, 1: this.feats, 2: B.F0, 3: B.EN, 4: fr.pw, 5: fr.pb, 6: B.XA }, Math.ceil(F / 64), WIDTH, 'frame expansion');
    block(fr.block, B.XA, B.XB, 0, 'frame block');
    linear(fr.prosody, B.XB, B.QKV, null, 'prosody');
    disp(this.k.sp, { 0: uni(F), 1: B.QKV }, Math.ceil(F / 256), 1, 'softplus f0');
    return p;
  }

  // One compute pass of a plan's calls, in order.
  runPass(enc, calls) {
    const pass = enc.beginComputePass();
    for (const c of calls) { pass.setPipeline(c.pipe); pass.setBindGroup(0, c.bg); pass.dispatchWorkgroups(c.gx, c.gy); }
    pass.end();
  }

  // The first `bytes` of a readback buffer (a MAP_READ buffer after a copy), as a float32 copy; the buffer is unmapped again.
  async mapFloats(buf, bytes) {
    await buf.mapAsync(GPUMapMode.READ, 0, bytes);
    try {
      return new Float32Array(buf.getMappedRange(0, bytes).slice(0));
    } finally {
      buf.unmap();
    }
  }

  // One push of a whole sentence: frames [384][T], f0 [T], energy [T] (Float32Arrays), written and submitted in one compute pass.
  // Returns { T, latents } where latents is the [512][T] buffer (stride T) that the next run overwrites. With read, the latents are also copied to the readback buffer, and read() maps it: a run made with read must be read before the next
  // one is made with read. frames = null: the frames are the ones frames(T) left in XB (the frame pass, no copy). That needs no other pass
  // since: a text pass or a run overwrites XB, and clears framesFor, which this checks.
  run(frames, f0, energy, { read = false } = {}) {
    const T = f0.length, d = this.device;
    if (energy.length !== T || (frames !== null && frames.length !== WIDTH * T)) {
      throw new Error(`bad inputs for ${T} frames: frames ${frames === null ? 'in XB' : frames.length}, energy ${energy.length}`);
    }
    if (frames === null && this.framesFor !== T) throw new Error(`run(null): XB holds the frames of frames(${this.framesFor}), not of ${T}`);
    if (read && this.readPending) throw new Error('read the run in flight before another run is made with read');
    const p = this.plan(T);
    if (frames !== null) d.queue.writeBuffer(this.buf.FR, 0, frames);
    d.queue.writeBuffer(this.buf.F0, 0, f0);
    d.queue.writeBuffer(this.buf.EN, 0, energy);
    this.framesFor = null;   // the run's first dispatch reads FR (XB), and its blocks then overwrite XB
    const enc = d.createCommandEncoder();
    this.runPass(enc, p.calls);
    const out = { T, latents: this.buf.LAT, readBytes: 0 };
    if (read) {
      // The layout of the readback: the latents [512][T].
      enc.copyBufferToBuffer(this.buf.LAT, 0, this.readback, 0, 4 * LATENTS * T);
      out.readBytes = 4 * LATENTS * T;
      this.readPending = true;
    }
    d.queue.submit([enc.finish()]);
    return out;
  }

  // The text encoder for a sentence of n tokens (n <= 512): x0 [384][n] (textInput) goes to the arena's XA, the D text blocks and the duration
  // head run in one compute pass, and the features [384][n] stay in this.feats for the frame pass. Resolves to the durations (40 Hz units,
  // before the rounding), a Float32Array(n). A second call before this one resolves throws.
  async encode(x0, n) {
    if (this.textPending) throw new Error('encode: the last text pass is not read yet (await its encode first)');
    if (x0.length !== WIDTH * n) throw new Error(`bad input for ${n} tokens: ${x0.length} values, expected ${WIDTH * n}`);
    const p = this.textPlan(n), d = this.device;
    this.textPending = true;
    try {
      d.queue.writeBuffer(this.buf.XA, 0, x0);
      const enc = d.createCommandEncoder();
      this.runPass(enc, p.calls);
      enc.copyBufferToBuffer(this.durs, 0, this.dursRead, 0, 4 * n);
      d.queue.submit([enc.finish()]);
      this.textFor = n;
      this.framesFor = null;   // the text blocks overwrite XB
      return await this.mapFloats(this.dursRead, 4 * n);
    } finally {
      this.textPending = false;
    }
  }

  // The frame pass from the last encode's features: counts40 (Uint32Array of the n tokens' 40 Hz counts, after the
  // rounding) and F = 2 sum(counts40). Each token gives 2 counts40 frames with phase i / count; the expansion, the frame block and the prosody
  // run in one pass. Resolves to the raw F0 (after the softplus) and energy, Float32Array(F) each, before the worker's plan (zt_gpu_plan). The frames stay in
  // XB for run(null, ...). A second call before this one resolves throws.
  async frames(counts40, F) {
    const n = counts40.length, d = this.device;
    if (this.textFor !== n) throw new Error(`frames: the features are for ${this.textFor} tokens (the last encode), not ${n}`);
    let total = 0;
    for (let t = 0; t < n; t++) total += 2 * counts40[t];
    if (total !== F) throw new Error(`frames: the counts make ${total} frames, not ${F}`);
    if (this.framesPending) throw new Error('frames: the last frame pass is not read yet (await its frames first)');
    const p = this.framesPlan(F);
    this.framesPending = true;
    try {
      const tok = new Uint32Array(F), phase = new Float32Array(F);
      for (let t = 0, f = 0; t < n; t++) {
        const count = 2 * counts40[t];
        for (let i = 0; i < count; i++, f++) { tok[f] = t; phase[f] = Math.fround(i / count); }
      }
      d.queue.writeBuffer(this.buf.F0, 0, tok);
      d.queue.writeBuffer(this.buf.EN, 0, phase);
      d.queue.writeBuffer(p.nUni, 0, new Uint32Array([F, n, 0, 0]));
      const enc = d.createCommandEncoder();
      this.runPass(enc, p.calls, null);
      enc.copyBufferToBuffer(this.buf.QKV, 0, this.framesRead, 0, 8 * F);
      d.queue.submit([enc.finish()]);
      this.framesFor = F;
      const all = await this.mapFloats(this.framesRead, 8 * F);
      return { f0: all.subarray(0, F), energy: all.subarray(F, 2 * F) };
    } finally {
      this.framesPending = false;
    }
  }

  // The latents [512][T] of a run made with read: true. The readback buffer is
  // freed for the next run (nothing is freed per run).
  async read(out) {
    if (!out.readBytes) throw new Error('the run was not made with { read: true }, or was read already');
    const n = out.readBytes, rb = this.readback, T = out.T;
    let mapped = false, all = null;
    try {
      await rb.mapAsync(GPUMapMode.READ, 0, n);
      mapped = true;
      all = new Float32Array(rb.getMappedRange(0, n).slice(0));
    } finally {
      if (mapped) rb.unmap();
      this.readPending = false;
      out.readBytes = 0;
    }
    return { latents: all.subarray(0, LATENTS * T) };
  }

  // The bytes of the current arena and of its readback buffer (the worker's replies report them).
  memory() {
    return { frames: this.maxFrames, weights: this.weightBytes, arena: arenaBytes(this.maxFrames),
      readback: readbackBytes(this.maxFrames) };
  }

  destroy() {
    for (const p of this.plans.values()) p.uniforms.forEach((u) => u.destroy());
    this.keep.forEach((b) => b.destroy());
    this.arenaBufs.forEach((b) => b.destroy());
    this.plans.clear();
    this.keep = [];
    this.arenaBufs = [];
    this.readPending = false;
  }
}
