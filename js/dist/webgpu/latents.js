// The student's acoustic pre-pass on WebGPU : one push of a whole sentence from a fresh state, with the latents [512][T] left on the GPU. The weights come from the student part of the model, checked
// against the settings and shapes the kernels assume. The kernels are in latents-kernels.js and the Linears are the generator's tiled
// convolution (kernels.js, k = 1). Built like generator.js: a plan of calls per length with its uniforms and bind groups.
// The pre-pass is causal, so a sentence runs in chunks of CHUNK columns (the last one shorter), each one compute pass in call order and one
// queue submission, with the state a chunk hands to the next kept in small buffers: the six columns of history before each block's depthwise
// convolution, the running energy of each block's scan, and the delta-rule memory of each mixer. The activations are therefore the size of a chunk,
// fixed for every sentence (the chunk arena); only the sentence's latents [512][T] (which the generator reads by window, and the conditioner's
// statistics need whole) and its F0 and energy grow with the sentence (capacity). The latents are read back through two small buffers, a group at a time.
// The text encoder: textInput gives x0 [384][n]; encode runs the text blocks (the acoustic ConvNeXt block, with the depthwise
// context centred on the lookahead) and leaves the features [384][n] in FEATS, and reads back the durations (40 Hz units, before the
// rounding). The frame pass: frames(counts40, F) expands the features to F frames (XA), runs the frame block (in chunks, like the acoustic blocks) into
// the first rows of the latents buffer, where run(null, ...) takes the frames, and reads back the raw F0 (after the softplus) and energy.
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
// The capacities in frames of the sentence's buffers (latents, F0, energy): a growth goes to the first of these that holds the sentence (never beyond MAX_FRAMES).
export const BUCKETS = [600, 800, 1000, 1250, 1500, 1800, 2200, MAX_FRAMES];
export const MAX_TOKENS = 512;     // the most tokens of a sentence the text encoder takes
export const CHUNK = 512;          // the columns of a chunk, a multiple of 256 (the energy scan's pieces) and at least MAX_TOKENS (the text encoder runs in one piece)
export const COORDS = 4;           // the voice's length coordinates (voice_fold has 1 + COORDS rows of WIDTH)
export const HIST_COLS = KERNEL - 1; // the columns of history before a depthwise convolution
export const RING = 512 * 1024;    // bytes of each of the two readback buffers (the latents are read back in groups of this size)
// The chunk arena's buffers in words per column (a chunk of T columns uses the first T columns of each row). Values that are not alive together share
// a buffer, and no dispatch reads and writes one (see planner): in a block the depthwise output D is dead once the norm has read it, so the hidden layer H
// takes its place; the norm output N is dead once the up projection has read it, so the response H2 takes its place. In a mixer HN and GM are N (HN is dead after
// the gate Linear, GM is made by the head norm), GP is D, and YT is QKV (dead once the transposes have read it). The output Linear's rows (512 per column)
// go to QKV, which the last mixer has finished with. The mixer's reciprocal norms [t][12] (PREPARE) share CUM: a mixer does not otherwise use it,
// and the next block's energy scan writes it after the mixer.
export const PER_COLUMN = { XA: WIDTH, XB: WIDTH, D: WIDTH, N: WIDTH, CUM: WIDTH, SC: 1, QKV: FEATURES, QKVT: FEATURES, G: DECAY, GT: DECAY };
// The sentence's buffers in words per frame: the latents [512][T] (stride T) and the F0 and energy [T]. The frame pass leaves the sentence's frames [384][T] in
// the first 384 rows of the latents buffer: a chunk's condition reads its columns of the frames before its output writes the same columns of the latents,
// and no other chunk's columns are touched. The latents are valid until the next frame pass or run is submitted (the worker reads a sentence's
// windows before it prepares the next sentence, so no window reads an overwritten run).
export const PER_FRAME = { LAT: LATENTS, F0: 1, EN: 1 };
export const WORDS_PER_COLUMN = Object.values(PER_COLUMN).reduce((a, b) => a + b, 0);
export const WORDS_PER_FRAME = Object.values(PER_FRAME).reduce((a, b) => a + b, 0);
// The bytes of the chunk arena (the same for every sentence), of the sentence's buffers for `frames` frames, and of the two readback buffers.
export const chunkBytes = 4 * WORDS_PER_COLUMN * CHUNK;
export const sentenceBytes = (frames) => 4 * WORDS_PER_FRAME * frames;
export const arenaBytes = (frames) => chunkBytes + sentenceBytes(frames);
export const readbackBytes = () => 2 * RING;
const STATE_BYTES = 4 * (BLOCKS + 1) * (WIDTH * HIST_COLS + WIDTH) + 4 * MIXER_AFTER.length * HEADS * HEAD_DIM * HEAD_DIM; // the history, energy and memory
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

// The most frames the acoustic pre-pass takes on this device: the latents [512][T] must fit a storage binding and a buffer (the dispatches are
// the size of a chunk, whatever the sentence's length).
export const frameLimit = (device) => {
  const l = device.limits;
  return Math.min(Math.floor(l.maxStorageBufferBindingSize / (4 * LATENTS)), Math.floor(l.maxBufferSize / (4 * LATENTS)));
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
  // An acoustic pre-pass for sentences of up to maxFrames frames (the capacity of its sentence buffers; ensureCapacity grows it).
  static async create(device, part, { maxFrames = MAX_FRAMES } = {}) {
    const b = new GpuAcoustic(device, part, maxFrames);
    await b.ready;
    await device.queue.onSubmittedWorkDone();
    return b;
  }

  constructor(device, part, maxFrames) {
    checkModel(part);
    if (!Number.isInteger(maxFrames) || maxFrames < 1) throw new Error(`maxFrames must be a positive integer, not ${maxFrames}`);
    if (maxFrames > MAX_FRAMES) throw new Error(`maxFrames ${maxFrames} is above MAX_FRAMES (${MAX_FRAMES}): the frame pass's readback is sized for it`);
    const limit = frameLimit(device);
    if (maxFrames > limit) throw new Error(`maxFrames ${maxFrames} is above this device's limit of ${limit} frames (the latents buffer of 512 words per frame)`);
    Object.assign(this, { device, part, maxFrames, keep: [], arenaBufs: [], plans: new Map(), kerns: new Map(), weightBytes: 0,
      readPending: false, failedFrames: null, textPending: false, textFor: null, framesPending: false, framesFor: null });
    // The recurrence and its prepare dispatch (PREPARE, before each mixer's recurrence).
    const rec = KN.RECURRENCE, prep = KN.PREPARE;
    this.k = {
      conv: this.kern(convWGSL({ k: 1, dil: 1, pad: 0, bias: true }, TILE), 'linear'),
      cond: this.kern(KN.CONDITION_CHUNK, 'condition'), dw: this.kern(KN.DEPTHWISE, 'depthwise'), ln: this.kern(KN.LAYERNORM, 'layer norm'),
      gelu: this.kern(KN.GELU, 'gelu'), scan: this.kern(KN.SCAN, 'energy scan'), scale: this.kern(KN.SCALE, 'response scale'),
      resp: this.kern(KN.RESPONSE, 'response'), tr: this.kern(KN.TRANSPOSE, 'transpose'), rec: this.kern(rec, 'delta rule'),
      prep: this.kern(prep, 'delta prep'),
      gnorm: this.kern(KN.GATE_NORM, 'head norm and gate'), sp: this.kern(KN.SOFTPLUS, 'softplus'),
      expand: this.kern(KN.EXPAND_CHUNK, 'frame expansion'),
      // The chunked forms (the acoustic blocks and the frame block; the text encoder runs in one piece and keeps the plain ones).
      dwc: this.kern(KN.DEPTHWISE_CHUNK, 'chunk depthwise'), hist: this.kern(KN.HISTORY, 'history'), place: this.kern(KN.PLACE, 'place'),
      scanc: this.kern(KN.SCAN_CHUNK, 'chunk energy scan'), recc: this.kern(KN.RECURRENCE_CHUNK, 'chunk delta rule'),
    };
    this.w = this.loadWeights();
    // The text encoder's features [384][n] (kept, not in the arena: the frame pass reads them) and its durations (4 rows of n; row 0 is
    // read back through dursRead). The raw F0 and energy of frames() come back through framesRead (rows of F, 2 * MAX_FRAMES words).
    this.feats = this.mk(4 * WIDTH * MAX_TOKENS, STORAGE);
    this.durs = this.mk(4 * 4 * MAX_TOKENS, STORAGE);
    this.dursRead = this.mk(4 * MAX_TOKENS, READBACK);
    this.framesRead = this.mk(4 * 2 * MAX_FRAMES, READBACK);
    // The chunk uniform { g0, f, n, 0 } (see latents-kernels.js) is written before each chunk's submission; the two readback buffers for the latents.
    this.cu = this.mk(16, UNIFORM);
    this.rings = [this.mk(RING, READBACK), this.mk(RING, READBACK)];
    this.allocChunk();
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

  // The student's tensors: every Linear for convWGSL, as the model file stores it (block-quantized) or packed as floats; the residual variant for
  // the Linears whose output adds the stream.
  loadWeights() {
    const P = this.part, up = (name) => this.upload(P.f32(name));
    const lin = (name, cout, cin, residual = false) => {
      const s = P.shape(`${name}.weight`);
      if (s[0] !== cout || s[1] !== cin) throw new Error(`${name}: shape [${s}], expected [${cout}, ${cin}]`);
      const pk = cout % 4 === 0 && cin % 8 === 0 ? P.packed(`${name}.weight`) : null;
      if (!pk) return { kern: residual ? this.kern(convWGSL({ k: 1, dil: 1, pad: 0, bias: true, residual }, TILE), 'linear + residual') : this.k.conv,
        w: this.upload(packLinear(P.f32(`${name}.weight`), cout, cin)), bias: up(`${name}.bias`), cin, cout };
      // Block-quantized in the model file: kept packed, the shader dequantizes the tile it loads (see convWGSL).
      if (pk.rows !== cout || pk.len !== cin || pk.words !== Math.ceil(Math.ceil((cout * cin * pk.bits) / 8) / 4)) {
        throw new Error(`${name}: packed shape [${pk.shape}], expected [${cout}, ${cin}]`);
      }
      return { kern: this.kern(convWGSL({ k: 1, dil: 1, pad: 0, bias: true, residual, packed: { bits: pk.bits, block: pk.block } }, TILE),
        residual ? 'linear + residual' : 'linear'), w: this.upload(pk.data), bias: up(`${name}.bias`), cin, cout };
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

  // The chunk arena (the same for every sentence, with the aliases of PER_COLUMN) and the state a chunk hands to the next, made once.
  allocChunk() {
    const A = (words) => this.mk(4 * words * CHUNK, STORAGE);
    const b = Object.fromEntries(Object.entries(PER_COLUMN).map(([name, words]) => [name, A(words)]));
    this.chunk = { ...b, H: b.D, H2: b.N, HN: b.N, GP: b.D, YT: b.QKV, GM: b.N };
    // Per block (the six acoustic blocks and the frame block): the six columns of history of its depthwise and the running energy of its scan;
    // per mixer: the delta-rule memory. None is initialised: a sentence's first chunk (g0 = 0) does not read them.
    const blockState = () => ({ hist: this.mk(4 * WIDTH * HIST_COLS, STORAGE), carry: this.mk(4 * WIDTH, STORAGE) });
    this.state = { blocks: range(BLOCKS).map(blockState), frame: blockState(),
      mem: Object.fromEntries(MIXER_AFTER.map((i) => [i, this.mk(4 * HEADS * HEAD_DIM * HEAD_DIM, STORAGE)])) };
  }

  // The sentence's buffers, sized for the current capacity (maxFrames): the latents (which also hold the frames, see PER_FRAME), the F0 and the energy.
  // Every buffer goes to arenaBufs, so a growth or destroy frees them all.
  allocArena() {
    const M = this.maxFrames, A = (words) => this.mk(4 * words * M, STORAGE, this.arenaBufs);
    this.buf = { ...this.chunk, LAT: A(LATENTS), F0: A(1), EN: A(1) };
  }

  // The capacity for a sentence of T frames: the first bucket that holds T, at most MAX_FRAMES and this device's limit.
  capacityFor(T) {
    const want = BUCKETS.find((b) => b >= T) ?? MAX_FRAMES;
    return Math.min(want, MAX_FRAMES, frameLimit(this.device));
  }

  // Destroys the sentence's buffers and the cached plans, then makes them for M frames in error scopes. The old ones go first: their memory is
  // needed for the new one. On failure no sentence buffers are left (the caller makes them again); the error has oom set when the device ran out of
  // memory.
  async remake(M) {
    for (const p of this.plans.values()) p.uniforms.forEach((u) => u.destroy());
    this.plans.clear();
    this.arenaBufs.forEach((b) => b.destroy());
    this.arenaBufs = [];
    this.maxFrames = M;
    this.framesFor = null;   // frames() left them in the latents buffer, which is destroyed here
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
      throw Object.assign(new Error(`the acoustic buffers for ${M} frames were not made (${oom ? `out of memory: ${oom.message}` : `validation: ${invalid.message}`})`),
        { oom: oom !== null });
    }
  }

  // Grows the sentence's buffers to the bucket that holds a sentence of T frames (capacityFor), when T is more than maxFrames. The old ones go first
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
      throw new Error(`the acoustic buffers for ${want} frames ran out of memory before; a sentence of ${T} frames runs in WebAssembly`);
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
        throw Object.assign(new Error(`the acoustic buffers could not be made again at ${old} frames after a failed growth to ${want} (${e2.message})`),
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

  // The plan of a chunk of tc columns of the acoustic blocks, of a chunk of the frame pass, and of the text encoder's n tokens (the text
  // encoder runs in one piece).
  plan(tc) {
    if (!Number.isInteger(tc) || tc < 1 || tc > CHUNK) throw new Error(`a chunk of ${tc} columns is outside 1..${CHUNK}`);
    return this.cached(tc, () => this.buildPlan(tc));
  }

  textPlan(n) {
    if (!Number.isInteger(n) || n < 1 || n > MAX_TOKENS) throw new Error(`n = ${n} tokens is outside 1..${MAX_TOKENS}`);
    return this.cached(`text:${n}`, () => this.buildTextPlan(n));
  }

  framesPlan(tc) {
    if (!Number.isInteger(tc) || tc < 1 || tc > CHUNK) throw new Error(`frames: a chunk of ${tc} columns is outside 1..${CHUNK}`);
    return this.cached(`frames:${tc}`, () => this.buildFramesPlan(tc));
  }

  // The builder of a plan of T columns: p.calls (each with its uniforms and bind groups) and the steps they are made of: a Linear, a ConvNeXt
  // block (the acoustic blocks, the text encoder's and the frame block: the same dispatches) and a mixer. A plan with state (st, mem) is a chunk
  // of a longer sentence; its kernels take the chunk's place from the chunk uniform this.cu.
  planner(T) {
    const B = this.buf, k = this.k, p = { T, calls: [], uniforms: [] }, cu = this.cu, made = new Map();
    const uni = (...v) => {
      const key = v.join(',');
      if (made.has(key)) return made.get(key);
      const u = this.device.createBuffer({ size: 16, usage: UNIFORM });
      this.device.queue.writeBuffer(u, 0, new Uint32Array([...v, 0, 0, 0].slice(0, 4)));
      p.uniforms.push(u);
      made.set(key, u);
      return u;
    };
    const disp = (kern, binds, gx, gy, label) => {
      const entries = Object.entries(binds).filter(([, b]) => b).map(([i, b]) => ({ binding: Number(i), resource: { buffer: b } }));
      p.calls.push({ pipe: kern.pipe, bg: this.device.createBindGroup({ layout: kern.pipe.getBindGroupLayout(0), entries }), gx, gy, label });
    };
    // A Linear: out[co][t] = bias[co] + sum_ci W[co][ci] in[ci][t] (+ res[co][t]).
    const linear = (l, src, dst, res, label) => disp(l.kern, { 0: uni(T, l.cin, l.cout, T), 1: src, 2: l.w, 3: dst, 4: l.bias, 5: res },
      Math.ceil(T / CONV_L), Math.ceil(l.cout / CONV_CO), label);
    // A ConvNeXt block from cur into dst = down(...) + cur. after: the depthwise's columns past each t (0 for the causal acoustic block). st: the
    // block's state when it runs in chunks (its depthwise reads the history, and the history is saved for the next chunk once the depthwise has read cur).
    const block = (blk, cur, dst, after, name, st = null) => {
      const W = WIDTH * T;
      if (st) {
        disp(k.dwc, { 0: uni(T), 1: cur, 2: blk.dw, 3: blk.dwB, 4: B.D, 5: st.hist, 6: cu }, Math.ceil(T / 64), WIDTH, `${name} depthwise`);
        disp(k.hist, { 0: uni(T), 1: cur, 2: st.hist }, Math.ceil((WIDTH * HIST_COLS) / 64), 1, `${name} history`);
      } else {
        disp(k.dw, { 0: uni(T, after), 1: cur, 2: blk.dw, 3: blk.dwB, 4: B.D }, Math.ceil(T / 64), WIDTH, `${name} depthwise`);
      }
      disp(k.ln, { 0: uni(T), 1: B.D, 2: blk.lnW, 3: blk.lnB, 4: B.N }, Math.ceil(T / 64), 1, `${name} norm`);
      linear(blk.up, B.N, B.H, null, `${name} up`);
      disp(k.gelu, { 0: uni(W), 1: B.H }, Math.ceil(W / 256), 1, `${name} gelu`);
      if (st) disp(k.scanc, { 0: uni(T), 1: B.H, 2: B.CUM, 3: st.carry, 4: cu }, WIDTH, 1, `${name} energy`);
      else disp(k.scan, { 0: uni(T), 1: B.H, 2: B.CUM }, WIDTH, 1, `${name} energy`);
      disp(k.scale, { 0: uni(T), 1: B.CUM, 2: B.SC }, Math.ceil(T / 64), 1, `${name} scale`);
      disp(k.resp, { 0: uni(T, W), 1: B.H, 2: B.CUM, 3: B.SC, 4: blk.gamma, 5: blk.beta, 6: B.H2 }, Math.ceil(W / 256), 1, `${name} response`);
      linear(blk.down, B.H2, dst, cur, `${name} down + residual`);
    };
    // The mixer: its inputs are read by earlier dispatches than the buffers they share with (HN = N is written after the block's up
    // projection has read N; GP = D is free; YT = QKV is written after the transposes have read QKV; GM = N is written after the gate Linear has read HN, and is
    // read only by the output Linear). Its recurrence starts from the memory the last chunk left.
    const mixer = (i, X, OUT) => {
      const m = this.w.mixers[i], name = `mixer ${i}`;
      disp(k.ln, { 0: uni(T), 1: X, 2: m.lnW, 3: m.lnB, 4: B.HN }, Math.ceil(T / 64), 1, `${name} norm`);
      linear(m.qkv, B.HN, B.QKV, null, `${name} qkv`);
      linear(m.dec, B.HN, B.G, null, `${name} decay and write`);
      linear(m.gate, B.HN, B.GP, null, `${name} gate`);
      disp(k.tr, { 0: uni(T, FEATURES), 1: B.QKV, 2: B.QKVT }, Math.ceil(T / 32), Math.ceil(FEATURES / 32), `${name} transpose qkv`);
      disp(k.tr, { 0: uni(T, DECAY), 1: B.G, 2: B.GT }, Math.ceil(T / 32), Math.ceil(DECAY / 32), `${name} transpose gates`);
      disp(k.prep, { 0: uni(T), 1: B.QKVT, 2: B.GT, 3: B.CUM }, Math.ceil(T / 64), HEADS, `${name} delta prep`);
      disp(k.recc, { 0: uni(T), 1: B.QKVT, 2: B.GT, 3: B.YT, 4: B.CUM, 5: this.state.mem[i], 6: cu }, HEADS, 1, `${name} delta rule`);
      disp(k.gnorm, { 0: uni(T), 1: B.YT, 2: B.GP, 3: m.outNW, 4: m.outNB, 5: B.GM }, Math.ceil(T / 64), HEADS, `${name} head norm and gate`);
      linear(m.out, B.GM, OUT, X, `${name} out + residual`);
    };
    // The chunk's columns [r][T] into the sentence's rows [r][f] at the chunk's place.
    const place = (src, dst, rows, label) => disp(k.place, { 0: uni(T), 1: src, 2: dst, 3: cu }, Math.ceil(T / 64), rows, label);
    return { p, disp, uni, linear, block, mixer, place };
  }

  // A chunk of the acoustic blocks: the condition reads its columns of the frames (the first 384 rows of the latents buffer), the blocks and mixers run
  // with their state, and the output Linear's rows (in QKV) are placed in the latents buffer, over the frames' columns that the condition has read.
  buildPlan(T) {
    const { p, disp, uni, linear, block, mixer, place } = this.planner(T);
    const B = this.buf, st = this.state;
    // The condition: x = (bias + w0 f0 + w1 energy) + frames; then the blocks, with the mixers after blocks 1 and 4.
    disp(this.k.cond, { 0: uni(T), 1: B.LAT, 2: B.F0, 3: B.EN, 4: this.w.cond.cw, 5: this.w.cond.cb, 6: B.XA, 7: this.cu }, Math.ceil(T / 64), WIDTH, 'condition');
    let cur = B.XA;
    const other = (x) => (x === B.XA ? B.XB : B.XA);
    for (let i = 0; i < BLOCKS; i++) {
      const nxt = other(cur);
      block(this.w.blocks[i], cur, nxt, 0, `block ${i}`, st.blocks[i]);
      cur = nxt;
      if (MIXER_AFTER.includes(i)) {
        const out = other(cur);
        mixer(i, cur, out);
        cur = out;
      }
    }
    linear(this.w.out, cur, B.QKV, null, 'output');
    place(B.QKV, B.LAT, LATENTS, 'latents');
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

  // A chunk of the frame pass: the expansion (XA) of the chunk's frames, the frame block (XA -> XB, with its state) placed into the first 384 rows of the
  // latents buffer (where run(null, ...) reads them), the prosody Linear (XB -> QKV, rows 0 and 1 of the chunk) and the softplus of row 0 in place.
  // Aliases, each safe because the pass reads an input before anything writes it: the expansion's token index and phase are read from F0 and EN, which
  // run() fills only after this pass (frames() writes them just before); the prosody output goes to QKV, which the mixers write only inside run().
  // The plan depends on the chunk's length alone; the expansion reads the features' stride n from the chunk uniform.
  buildFramesPlan(T) {
    const { p, disp, uni, linear, block, place } = this.planner(T);
    const B = this.buf, fr = this.w.frames;
    disp(this.k.expand, { 0: uni(T), 1: this.feats, 2: B.F0, 3: B.EN, 4: fr.pw, 5: fr.pb, 6: B.XA, 7: this.cu }, Math.ceil(T / 64), WIDTH, 'frame expansion');
    block(fr.block, B.XA, B.XB, 0, 'frame block', this.state.frame);
    place(B.XB, B.LAT, WIDTH, 'frames');
    linear(fr.prosody, B.XB, B.QKV, null, 'prosody');
    disp(this.k.sp, { 0: uni(T), 1: B.QKV }, Math.ceil(T / 256), 1, 'softplus f0');
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

  // One push of a whole sentence: frames [384][T], f0 [T], energy [T] (Float32Arrays), written and then run in chunks of CHUNK columns, one submission each.
  // Returns { T, latents } where latents is the [512][T] buffer (stride T) that the next frame pass or run overwrites. With read, the latents are also copied to the
  // readback buffers, and read() maps them in groups: a run made with read must be read before the next one is made with read. frames = null: the frames are
  // the ones frames(T) left in the latents buffer (the frame pass, no copy). That needs no other pass since: a text pass or a run overwrites them,
  // and clears framesFor, which this checks.
  run(frames, f0, energy, { read = false } = {}) {
    const T = f0.length, d = this.device;
    if (energy.length !== T || (frames !== null && frames.length !== WIDTH * T) || T > this.maxFrames) {
      throw new Error(`bad inputs for ${T} frames (capacity ${this.maxFrames}): frames ${frames === null ? 'in the latents buffer' : frames.length}, energy ${energy.length}`);
    }
    if (frames === null && this.framesFor !== T) throw new Error(`run(null): the latents buffer holds the frames of frames(${this.framesFor}), not of ${T}`);
    if (read && this.readPending) throw new Error('read the run in flight before another run is made with read');
    if (frames !== null) d.queue.writeBuffer(this.buf.LAT, 0, frames);
    d.queue.writeBuffer(this.buf.F0, 0, f0);
    d.queue.writeBuffer(this.buf.EN, 0, energy);
    this.framesFor = null;   // the first chunk's condition reads the frames, and the chunks' outputs then overwrite them
    const out = { T, latents: this.buf.LAT, readBytes: 0 };
    for (let g0 = 0; g0 < T; g0 += CHUNK) {
      const tc = Math.min(CHUNK, T - g0), p = this.plan(tc), last = g0 + tc === T;
      d.queue.writeBuffer(this.cu, 0, new Uint32Array([g0, T, 0, 0]));
      const enc = d.createCommandEncoder();
      this.runPass(enc, p.calls);
      if (last && read) {
        // The layout of the readback: the latents [512][T], in groups of RING bytes; the first two groups go to the readback buffers now.
        out.readBytes = 4 * LATENTS * T;
        for (let k = 0; k < Math.min(2, this.groups(out.readBytes)); k++) this.copyGroup(enc, out.readBytes, k);
        this.readPending = true;
      }
      d.queue.submit([enc.finish()]);
    }
    return out;
  }

  // The groups of RING bytes that `bytes` of latents are read back in, the size of group k, and the copy of group k to its readback buffer.
  groups(bytes) { return Math.ceil(bytes / RING); }

  groupBytes(bytes, k) { return Math.min(RING, bytes - k * RING); }

  copyGroup(enc, bytes, k) { enc.copyBufferToBuffer(this.buf.LAT, k * RING, this.rings[k & 1], 0, this.groupBytes(bytes, k)); }

  // The text encoder for a sentence of n tokens (n <= 512): x0 [384][n] (textInput) goes to the chunk arena's XA, the D text blocks and the duration
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
      this.framesFor = null;   // a new text pass is a new sentence: its frame pass comes next
      return await this.mapFloats(this.dursRead, 4 * n);
    } finally {
      this.textPending = false;
    }
  }

  // The frame pass from the last encode's features: counts40 (Uint32Array of the n tokens' 40 Hz counts, after the
  // rounding) and F = 2 sum(counts40). Each token gives 2 counts40 frames with phase i / count; the expansion, the frame block and the prosody
  // run in chunks of CHUNK columns (one submission each). Resolves to the raw F0 (after the softplus) and energy, Float32Array(F) each, before the worker's plan (zt_gpu_plan). The frames stay in
  // the latents buffer for run(null, ...). A second call before this one resolves throws.
  async frames(counts40, F) {
    const n = counts40.length, d = this.device;
    if (this.textFor !== n) throw new Error(`frames: the features are for ${this.textFor} tokens (the last encode), not ${n}`);
    let total = 0;
    for (let t = 0; t < n; t++) total += 2 * counts40[t];
    if (total !== F) throw new Error(`frames: the counts make ${total} frames, not ${F}`);
    if (F > this.maxFrames) throw new Error(`frames: F = ${F} is outside 1..${this.maxFrames}`);
    if (this.framesPending) throw new Error('frames: the last frame pass is not read yet (await its frames first)');
    this.framesPending = true;
    try {
      const tok = new Uint32Array(F), phase = new Float32Array(F);
      for (let t = 0, f = 0; t < n; t++) {
        const count = 2 * counts40[t];
        for (let i = 0; i < count; i++, f++) { tok[f] = t; phase[f] = Math.fround(i / count); }
      }
      d.queue.writeBuffer(this.buf.F0, 0, tok);
      d.queue.writeBuffer(this.buf.EN, 0, phase);
      this.framesFor = null;
      for (let g0 = 0; g0 < F; g0 += CHUNK) {
        const tc = Math.min(CHUNK, F - g0), p = this.framesPlan(tc);
        d.queue.writeBuffer(this.cu, 0, new Uint32Array([g0, F, n, 0]));
        const enc = d.createCommandEncoder();
        this.runPass(enc, p.calls);
        // Rows 0 and 1 of the chunk's prosody output (F0, energy) to the same rows of the F columns read back.
        enc.copyBufferToBuffer(this.buf.QKV, 0, this.framesRead, 4 * g0, 4 * tc);
        enc.copyBufferToBuffer(this.buf.QKV, 4 * tc, this.framesRead, 4 * (F + g0), 4 * tc);
        d.queue.submit([enc.finish()]);
      }
      this.framesFor = F;
      const all = await this.mapFloats(this.framesRead, 8 * F);
      return { f0: all.subarray(0, F), energy: all.subarray(F, 2 * F) };
    } finally {
      this.framesPending = false;
    }
  }

  // The latents [512][T] of a run made with read: true, read in groups of RING bytes through the two readback buffers: the first two groups were copied
  // by the run, and both are mapped at once; as each group is read out, the group two after it is copied into its buffer and mapped, while the next group
  // is read. The buffers are free for the next run once this has resolved.
  async read(out) {
    if (!out.readBytes) throw new Error('the run was not made with { read: true }, or was read already');
    const bytes = out.readBytes, T = out.T, n = this.groups(bytes), all = new Float32Array(LATENTS * T);
    const maps = [null, null];
    const map = (k) => {
      const p = this.rings[k & 1].mapAsync(GPUMapMode.READ, 0, this.groupBytes(bytes, k));
      p.catch(() => {});   // a failure is thrown where the group is awaited; a group never awaited must not be an unhandled rejection
      maps[k & 1] = p;
    };
    try {
      map(0);
      if (n > 1) map(1);
      for (let k = 0; k < n; k++) {
        const rb = this.rings[k & 1], size = this.groupBytes(bytes, k);
        await maps[k & 1];
        all.set(new Float32Array(rb.getMappedRange(0, size)), (k * RING) / 4);
        rb.unmap();
        if (k + 2 < n) {
          const enc = this.device.createCommandEncoder();
          this.copyGroup(enc, bytes, k + 2);
          this.device.queue.submit([enc.finish()]);
          map(k + 2);
        }
      }
    } finally {
      for (const rb of this.rings) { try { rb.unmap(); } catch { /* not mapped */ } }
      this.readPending = false;
      out.readBytes = 0;
    }
    return { latents: all };
  }

  // The bytes of the chunk arena, of the sentence's buffers at the current capacity, of the state and of the readback buffers (the worker's replies report them).
  memory() {
    return { frames: this.maxFrames, weights: this.weightBytes, arena: chunkBytes, state: STATE_BYTES, sentence: sentenceBytes(this.maxFrames),
      readback: readbackBytes() };
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
