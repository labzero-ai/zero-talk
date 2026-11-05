// The generator on WebGPU: the network after the harmonic source, with its inverse STFT. Static per-site affines (set per utterance
// by setUtterance); a plan of calls per window length, all on activation buffers shared by every length. The kernels are in kernels.js, the window
// geometry in windows.js.
import { convWGSL, transposedVecWGSL, elementwiseWGSL, polarWGSL, istftFrameWGSL, istftOlaWGSL, TILE, STRIDED } from './kernels.js';
import { SAMPLES_PER_FRAME, CORE, HALO } from './windows.js';

// The buffer usage flags, as the WebGPU spec fixes them, so this module also loads where WebGPU is missing (Node.js, some
// browsers); it is only used where it exists.
const GPUBufferUsage = globalThis.GPUBufferUsage ?? { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 };
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const READBACK = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
export const OUTSTANDING = 4; // windows in flight (submitted, not yet read), each with its own readback buffer, read in order
const MAX_PLANS = 4;           // cached plans (calls and uniforms, one per window length); the least recently used one is destroyed
const DEFAULT_MAX_FRAMES = CORE + 2 * HALO; // 84: a 60-frame core and its halo on both sides (the worker grows the arena for 120)
const DIL = [1, 3, 5];
// A window's latents from the acoustic buffer (submit's {lat, T, s, n} form): out[c][t] = lat[c T + s + t] for t < len, where lat is
// [512][T] (stride T) and the window's input is [512][len] (stride len). One thread per (t, c).
const GATHER = `
struct P { len: u32, lin: u32, s: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> lat: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x; let c = g.y;
  if (t >= p.len) { return; }
  o[c * p.len + t] = lat[c * p.lin + p.s + t];
}`;
// The eight residual blocks: name, kernel size, channels, and whether the block runs on the 10 L grid (else 60 L + 1).
const BLOCKS = [
  ['noise_res.0', 7, 256, true], ['noise_res.1', 11, 128, false],
  ['resblocks.0', 3, 256, true], ['resblocks.1', 7, 256, true], ['resblocks.2', 11, 256, true],
  ['resblocks.3', 3, 128, false], ['resblocks.4', 7, 128, false], ['resblocks.5', 11, 128, false],
];

// PyTorch's Conv1d weight [cout][cin][k], packed for convWGSL as [cinPad][k][coutPad], zero outside the layer.
const packConv = (w, cout, cin, k, coutPad = cout, cinPad = cin) => {
  const out = new Float32Array(cinPad * k * coutPad);
  for (let co = 0; co < cout; co++) for (let ci = 0; ci < cin; ci++) for (let j = 0; j < k; j++) out[(ci * k + j) * coutPad + co] = w[(co * cin + ci) * k + j];
  return out;
};
// PyTorch's ConvTranspose1d weight [cin][cout][k], packed for transposedVecWGSL as [cin][k][cout].
const packTransposed = (w, cin, cout, k) => {
  const out = new Float32Array(cin * k * cout);
  for (let ci = 0; ci < cin; ci++) for (let co = 0; co < cout; co++) for (let j = 0; j < k; j++) out[(ci * k + j) * cout + co] = w[(ci * cout + co) * k + j];
  return out;
};
const padTo = (v, n) => { const out = new Float32Array(n); out.set(v); return out; };
// The inverse DFT's tables for 20 points: cs[k * 40 + n] = cos(2 pi k n / 20), cs[k * 40 + 20 + n] = sin(...), k < 10. Computed
// in float64, then rounded to float32.
const dftTable = () => {
  const out = new Float32Array(400);
  for (let k = 0; k < 10; k++) for (let n = 0; n < 20; n++) {
    const a = (2 * Math.PI * (k * n)) / 20;
    out[k * 40 + n] = Math.fround(Math.cos(a));
    out[k * 40 + 20 + n] = Math.fround(Math.sin(a));
  }
  return out;
};

export class GpuGenerator {
  // The weights, the pipelines, the shared activations and the audio readback are made here; a plan is made on the first window
  // of each length.
  static async create(device, model, { maxFrames = DEFAULT_MAX_FRAMES } = {}) {
    const g = new GpuGenerator(device, model, maxFrames);
    await g.ready;
    await device.queue.onSubmittedWorkDone();
    return g;
  }

  constructor(device, model, maxFrames) {
    if (!Number.isInteger(maxFrames) || maxFrames < 1) throw new Error(`maxFrames must be a positive integer, not ${maxFrames}`);
    Object.assign(this, {
      device, maxFrames, part: model.part('generator'), keep: [], plans: new Map(), kerns: new Map(), sites: new Map(),
      blocks: new Map(), hasAffines: false, tickets: [], seq: 0, arenaBufs: [],
    });
    const T = (n) => this.part.f32(n), S = (n) => this.part.shape(n);
    // Kernels that are not per layer.
    this.elem = {
      add: this.kern(elementwiseWGSL('xv + yv'), 'elementwise add'),
      addReflect: this.kern(elementwiseWGSL('xv + yv', true), 'elementwise add reflect'),
      avg3: this.kern(elementwiseWGSL('(xv + yv) / 3.0'), 'elementwise avg3'),
      leaky01: this.kern(elementwiseWGSL('select(0.1 * xv, xv, xv > 0.0)'), 'elementwise leaky 0.1'),
      leaky001: this.kern(elementwiseWGSL('select(0.01 * xv, xv, xv > 0.0)'), 'elementwise leaky 0.01'),
    };
    this.kPolar = this.kern(polarWGSL, 'polar');
    this.kFrame = this.kern(istftFrameWGSL, 'istft frame');
    this.kOla = this.kern(istftOlaWGSL, 'istft ola');
    this.kGather = this.kern(GATHER, 'gather latents');
    this.gatherU = this.mk(16, UNIFORM);
    this.cs = this.upload(dftTable());
    this.win = this.upload(T('stft.window'));
    // The network's layers outside the blocks.
    const r0 = S('ups.0.first.weight')[0];
    this.nc0 = this.op({ k: 12, dil: 1, stride: 6, pad: 3, bias: true }, STRIDED, packConv(T('noise_convs.0.weight'), 256, 22, 12, 256, 24), T('noise_convs.0.bias'), 24, 256);
    this.nc1 = this.op({ k: 1, dil: 1, bias: true }, TILE, packConv(T('noise_convs.1.weight'), 128, 22, 1, 128, 24), T('noise_convs.1.bias'), 24, 128);
    this.up0a = this.op({ k: 1, dil: 1 }, TILE, packConv(T('ups.0.first.weight'), r0, 512, 1), null, 512, r0);
    this.up0b = this.tOp({ k: 20, stride: 10, pad: 5, bias: false }, packTransposed(T('ups.0.core.weight'), r0, r0, 20), null, r0, r0);
    this.up0c = this.op({ k: 1, dil: 1, bias: true }, TILE, packConv(T('ups.0.last.weight'), 256, r0, 1), T('ups.0.last.bias'), r0, 256);
    this.up1 = this.tOp({ k: 12, stride: 6, pad: 3, bias: true }, packTransposed(T('ups.1.weight'), 256, 128, 12), T('ups.1.bias'), 256, 128);
    this.post = this.op({ k: 7, dil: 1, pad: 3, bias: true }, TILE, packConv(T('conv_post.weight'), 22, 128, 7, 24, 128), padTo(T('conv_post.bias'), 24), 128, 24);
    // The eight residual blocks, then the buffers shared by every window length.
    for (const [name, k, C, short] of BLOCKS) this.blocks.set(name, this.makeBlock(name, k, C, short));
    this.allocArena();
    this.ready = Promise.all([...this.kerns.values()].map(async (k) => {
      k.pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: k.code }), entryPoint: 'main' } });
    }));
  }

  // Buffers are made with mk: those that live as long as the generator go to `keep`, a plan's uniforms to the plan's list.
  mk(bytes, usage, list = this.keep) {
    const b = this.device.createBuffer({ size: Math.max(16, bytes), usage });
    list.push(b);
    return b;
  }

  upload(data) {
    const b = this.mk(data.byteLength, STORAGE);
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  // A kernel (one shader, one pipeline) keyed by its code, so that equal kernels are compiled once. The label names it in a plan's
  // calls; a convolution's label belongs to its operation instead (the code does not hold the channel counts).
  kern(code, label = null) {
    if (!this.kerns.has(code)) this.kerns.set(code, { code, label, pipe: null });
    return this.kerns.get(code);
  }

  // A convolution: its spec (convWGSL), tile, packed weights and bias, its channels and its label.
  op(spec, tile, w, bias, cin, cout) {
    const label = `conv k${spec.k} d${spec.dil} s${spec.stride ?? 1} ${cin}>${cout}`;
    return { kern: this.kern(convWGSL(spec, tile)), tile, w: this.upload(w), bias: bias ? this.upload(bias) : null, cin, cout, label,
      pro: !!spec.prologue, post: spec.residual ? 'res' : spec.accumulate ? 'acc' : null };
  }

  // A transposed convolution (transposedVecWGSL's spec), its packed weights and bias, its channels and its label.
  tOp(spec, w, bias, cin, cout) {
    return { kern: this.kern(transposedVecWGSL(spec)), w: this.upload(w), bias: bias ? this.upload(bias) : null, cin, cout,
      label: `tconv k${spec.k} s${spec.stride} ${cin}>${cout}` };
  }

  // Conv1d `p`: dense (p.weight, p.bias) or Tucker (p.first, p.core, p.last). pro: the input
  // prologue (the site's affine and snake); post: 'res' (add the residual passed by the caller) or 'acc' (add the stream in place).
  convLayer(p, { k, dil, pad }, { pro = false, post = null }) {
    const part = this.part, P = (s) => `${p}.${s}`;
    if (part.has(P('weight'))) {
      const [cout, cin, kk] = part.shape(P('weight'));
      if (kk !== k) throw new Error(`${p}: kernel ${kk}, expected ${k}`);
      const bias = part.has(P('bias'));
      return { dense: this.op({ k, dil, pad, prologue: pro, bias, residual: post === 'res', accumulate: post === 'acc' }, TILE,
        packConv(part.f32(P('weight')), cout, cin, k), bias ? part.f32(P('bias')) : null, cin, cout) };
    }
    const [r, cin] = part.shape(P('first.weight')), [cout] = part.shape(P('last.weight'));
    if (part.shape(P('core.weight'))[2] !== k) throw new Error(`${p}: core kernel is not ${k}`);
    return {
      first: this.op({ k: 1, dil: 1, pad: 0, prologue: pro, bias: false }, TILE, packConv(part.f32(P('first.weight')), r, cin, 1), null, cin, r),
      core: this.op({ k, dil, pad, prologue: false, bias: false }, TILE, packConv(part.f32(P('core.weight')), r, r, k), null, r, r),
      last: this.op({ k: 1, dil: 1, pad: 0, prologue: false, bias: true, residual: post === 'res', accumulate: post === 'acc' }, TILE,
        packConv(part.f32(P('last.weight')), cout, r, 1), part.f32(P('last.bias')), r, cout),
    };
  }

  // The static affine of one AdaIN site: its buffers (pro: mean, scale, shift, alpha per channel; ia: 1 / alpha).
  site(block, which, u, C) {
    const name = `${block}.adain${which}.${u}`, alpha = this.part.f32(`${block}.alpha${which}.${u}`);
    if (alpha.length !== C) throw new Error(`${name}: alpha has ${alpha.length} channels, expected ${C}`);
    const s = { name, alpha, pro: this.mk(16 * C, STORAGE), ia: this.mk(4 * C, STORAGE) };
    this.sites.set(name, s);
    return s;
  }

  // AdaINResBlock1: three units, each x = conv2(snake(adain2(conv1(snake(adain1(x)))))) + x.
  makeBlock(name, k, C, short) {
    const units = [];
    for (let u = 0; u < 3; u++) {
      const d = DIL[u];
      units.push({
        s1: this.site(name, 1, u, C), s2: this.site(name, 2, u, C),
        c1: this.convLayer(`${name}.convs1.${u}`, { k, dil: d, pad: (d * (k - 1)) / 2 }, { pro: true }),
        c2: this.convLayer(`${name}.convs2.${u}`, { k, dil: 1, pad: (k - 1) / 2 }, { pro: true, post: u === 1 ? 'acc' : 'res' }),
      });
    }
    const rank = Math.max(0, ...units.flatMap((un) => [un.c1, un.c2]).filter((c) => c.first).map((c) => c.first.cout));
    return { name, C, short, units, rank };
  }

  // The activations and the audio readback, shared by every window length and sized for the longest one. Every kernel indexes by
  // its length and channel uniforms, never by the buffer's size, so a shorter window uses the front of each buffer. A(words) makes
  // a buffer of float32 words. The readback ring takes the audio of OUTSTANDING windows in flight (see submit).
  allocArena() {
    const M = this.maxFrames, L16 = 10 * M, LH = 60 * M + 1, L1 = 60 * M, N = SAMPLES_PER_FRAME * M, r0 = this.up0a.cout;
    const blocks = [...this.blocks.values()], lenOf = (b) => (b.short ? L16 : LH);
    const maxIn = Math.max(...blocks.map((b) => b.C * lenOf(b))), maxT = Math.max(1, ...blocks.map((b) => b.rank * lenOf(b)));
    const A = (words) => this.mk(4 * words, STORAGE, this.arenaBufs);
    this.arena = {
      S: { in: A(maxIn), y: A(maxIn), t1: A(maxT), t2: A(maxT) },
      xin: A(512 * M), harin: A(24 * LH),
      nc0: A(256 * L16), xs0: A(256 * L16), nc1: A(128 * LH), xs1: A(128 * LH), lx: A(512 * M), u0a: A(r0 * M), u0b: A(r0 * L16),
      u0: A(256 * L16), x1: A(256 * L16), o: [A(256 * L16), A(256 * L16), A(256 * L16)],
      s0: A(256 * L16), st0: A(256 * L16), lx2: A(256 * L16), u1: A(128 * L1), x3: A(128 * LH),
      o3: [A(128 * LH), A(128 * LH), A(128 * LH)], s1: A(128 * LH), x4: A(128 * LH), lx4: A(128 * LH),
      post: A(24 * LH), pol: A(22 * LH), frames: A(20 * LH), audio: A(N),
    };
    this.staging = Array.from({ length: OUTSTANDING }, () => this.mk(4 * N, READBACK, this.arenaBufs));
  }

  // Grows the arena (the activations and the readback ring) to windows of maxFrames frames, when that is more than now; the weights and the
  // pipelines stay. The old arena and the cached plans are destroyed first (remake), then the new one is made in error scopes. When that
  // fails, the previous size is made again and a clear error is thrown (oom: out of memory); the generator keeps working at the old size.
  // When even the previous size cannot be made, the error is fatal (fatal = true).
  // Needs no window in flight (the worker calls it at the start of a speak).
  async ensureCapacity(maxFrames) {
    if (!Number.isInteger(maxFrames) || maxFrames < 1) throw new Error(`maxFrames must be a positive integer, not ${maxFrames}`);
    if (maxFrames <= this.maxFrames) return;
    if (this.tickets.length !== 0) throw new Error('read the outstanding windows before ensureCapacity');
    const old = this.maxFrames;
    try {
      await this.remake(maxFrames);
    } catch (e) {
      try {
        await this.remake(old);
      } catch (e2) {
        throw Object.assign(new Error(`the GPU arena could not be made again at ${old} frames after a failed growth to ${maxFrames} (${e2.message})`),
          { fatal: true });
      }
      throw Object.assign(new Error(`the GPU arena for ${maxFrames} frames was not made, the generator keeps ${old} frames (${e.message})`),
        { oom: e.oom === true });
    }
  }

  // Destroys the arena and the cached plans, then makes the arena for M frames in error scopes (the allocation is thrown here, before any
  // window is submitted). On failure no arena is left; ensureCapacity makes the previous one again.
  async remake(M) {
    for (const p of this.plans.values()) p.bufs.forEach((b) => b.destroy());
    this.plans.clear();
    this.arenaBufs.forEach((b) => b.destroy());
    this.arenaBufs = [];
    this.maxFrames = M;
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
      throw Object.assign(new Error(`${oom ? `out of memory: ${oom.message}` : `validation: ${invalid.message}`}`), { oom: oom !== null });
    }
  }

  // Sets the utterance's affines: a Map site name -> { mean, scale, shift } (Float32Array, one value per channel).
  setUtterance(affines) {
    for (const s of this.sites.values()) {
      const a = affines.get(s.name);
      if (!a) throw new Error(`no affine for ${s.name}`);
      const C = s.alpha.length, pro = new Float32Array(4 * C), ia = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        pro.set([a.mean[c], a.scale[c], a.shift[c], s.alpha[c]], 4 * c);
        ia[c] = Math.fround(1 / s.alpha[c]);
      }
      this.device.queue.writeBuffer(s.pro, 0, pro);
      this.device.queue.writeBuffer(s.ia, 0, ia);
    }
    this.hasAffines = true;
  }

  // A dispatch: its bind group (binding index -> buffer), its grid, and the label it is reported by.
  disp(p, kern, binds, gx, gy = 1, label = kern.label) {
    const entries = Object.entries(binds).filter(([, b]) => b).map(([i, b]) => ({ binding: +i, resource: { buffer: b } }));
    p.calls.push({ pipe: kern.pipe, bg: this.device.createBindGroup({ layout: kern.pipe.getBindGroupLayout(0), entries }), gx, gy, label });
  }

  uni(p, ...v) {
    const u = this.mk(16, UNIFORM, p.bufs);
    this.device.queue.writeBuffer(u, 0, new Uint32Array([...v, 0, 0, 0, 0].slice(0, 4)));
    return u;
  }

  conv(p, op, { src, dst, len, lin, pro = null, res = null }) {
    this.disp(p, op.kern, {
      0: this.uni(p, len, op.cin, op.cout, lin), 1: src, 2: op.w, 3: dst, 4: op.bias,
      5: op.post === 'res' ? res : null, 6: op.pro ? pro.pro : null, 7: op.pro ? pro.ia : null,
    }, Math.ceil(len / (op.tile.TX * op.tile.L_PT)), Math.ceil(op.cout / (op.tile.TY * op.tile.CO_PT)), op.label);
  }

  tconv(p, op, { src, dst, len, lin }) {
    this.disp(p, op.kern, { 0: this.uni(p, len, op.cin, op.cout, lin), 1: src, 2: op.w, 3: dst, 4: op.bias }, Math.ceil(len / 64), op.cout / 8, op.label);
  }

  ew(p, kern, x, y, o, channels, len, lin) {
    this.disp(p, kern, { 0: this.uni(p, len, lin), 1: x, 2: y, 3: o }, Math.ceil(len / 256), channels);
  }

  // One residual block from `input` into `out`: the stream lives in the shared scratch (S), the block's inputs are read where they
  // are. Unit 0 adds the input as its residual, unit 1 accumulates into the stream, unit 2 writes `out`.
  block(p, b, input, out, len) {
    const S = this.arena.S;
    b.units.forEach((un, u) => {
      const stream = u === 0 ? input : S.in;
      if (un.c1.dense) this.conv(p, un.c1.dense, { src: stream, dst: S.y, len, lin: len, pro: un.s1 });
      else {
        this.conv(p, un.c1.first, { src: stream, dst: S.t1, len, lin: len, pro: un.s1 });
        this.conv(p, un.c1.core, { src: S.t1, dst: S.t2, len, lin: len });
        this.conv(p, un.c1.last, { src: S.t2, dst: S.y, len, lin: len });
      }
      const dst = u === 2 ? out : S.in, res = u === 2 ? S.in : input;
      if (un.c2.dense) this.conv(p, un.c2.dense, { src: S.y, dst, len, lin: len, pro: un.s2, res });
      else {
        this.conv(p, un.c2.first, { src: S.y, dst: S.t1, len, lin: len, pro: un.s2 });
        this.conv(p, un.c2.core, { src: S.t1, dst: S.t2, len, lin: len });
        this.conv(p, un.c2.last, { src: S.t2, dst, len, lin: len, res });
      }
    });
  }

  // The calls of a run with L latent frames (L = 80 Hz frames, SAMPLES_PER_FRAME L output samples), on the shared buffers.
  buildPlan(L) {
    const A = this.arena, L16 = 10 * L, LH = 60 * L + 1, N = SAMPLES_PER_FRAME * L, p = { L, calls: [], bufs: [] };
    const B = (name, input, out, len) => this.block(p, this.blocks.get(name), input, out, len);
    // The source's convolutions, then the residual blocks of the two noise branches.
    this.conv(p, this.nc0, { src: A.harin, dst: A.nc0, len: L16, lin: LH });
    B('noise_res.0', A.nc0, A.xs0, L16);
    this.conv(p, this.nc1, { src: A.harin, dst: A.nc1, len: LH, lin: LH });
    B('noise_res.1', A.nc1, A.xs1, LH);
    // Step 0: ups.0 (Tucker), the sum with the source, the mean of resblocks 0 to 2.
    this.ew(p, this.elem.leaky01, A.xin, A.xin, A.lx, 512, L, L);
    this.conv(p, this.up0a, { src: A.lx, dst: A.u0a, len: L, lin: L });
    this.tconv(p, this.up0b, { src: A.u0a, dst: A.u0b, len: L16, lin: L });
    this.conv(p, this.up0c, { src: A.u0b, dst: A.u0, len: L16, lin: L16 });
    this.ew(p, this.elem.add, A.u0, A.xs0, A.x1, 256, L16, L16);
    ['resblocks.0', 'resblocks.1', 'resblocks.2'].forEach((n, i) => B(n, A.x1, A.o[i], L16));
    this.ew(p, this.elem.add, A.o[0], A.o[1], A.s0, 256, L16, L16);
    this.ew(p, this.elem.avg3, A.s0, A.o[2], A.st0, 256, L16, L16);
    // Step 1: ups.1, the reflection-padded sum with the source, the mean of resblocks 3 to 5, conv_post, the iSTFT.
    this.ew(p, this.elem.leaky01, A.st0, A.st0, A.lx2, 256, L16, L16);
    this.tconv(p, this.up1, { src: A.lx2, dst: A.u1, len: 60 * L, lin: L16 });
    this.ew(p, this.elem.addReflect, A.u1, A.xs1, A.x3, 128, LH, 60 * L);
    ['resblocks.3', 'resblocks.4', 'resblocks.5'].forEach((n, i) => B(n, A.x3, A.o3[i], LH));
    this.ew(p, this.elem.add, A.o3[0], A.o3[1], A.s1, 128, LH, LH);
    this.ew(p, this.elem.avg3, A.s1, A.o3[2], A.x4, 128, LH, LH);
    this.ew(p, this.elem.leaky001, A.x4, A.x4, A.lx4, 128, LH, LH);
    this.conv(p, this.post, { src: A.lx4, dst: A.post, len: LH, lin: LH });
    this.disp(p, this.kPolar, { 0: this.uni(p, 22 * LH, LH), 1: A.post, 2: A.pol }, Math.ceil((22 * LH) / 256));
    this.disp(p, this.kFrame, { 0: this.uni(p, LH), 1: A.pol, 2: this.cs, 3: this.win, 4: A.frames }, Math.ceil(LH / 64));
    this.disp(p, this.kOla, { 0: this.uni(p, N, LH), 1: A.frames, 2: this.win, 3: A.audio }, Math.ceil(N / 64));
    // Rows 22 and 23 of harin are the source's channel padding (zero weights in nc0 and nc1); every window clears them, since the
    // shared buffer may hold an earlier window's data there.
    return Object.assign(p, { xin: A.xin, harin: A.harin, audio: A.audio, pad: { offset: 4 * 22 * LH, size: 4 * 2 * LH } });
  }

  // The plan of window length L, built on first use; the least recently used plan is destroyed beyond MAX_PLANS.
  plan(L) {
    if (L > this.maxFrames) throw new Error(`a window of ${L} frames is longer than the generator's maxFrames (${this.maxFrames})`);
    if (this.plans.has(L)) { const p = this.plans.get(L); this.plans.delete(L); this.plans.set(L, p); return p; }
    while (this.plans.size >= MAX_PLANS) {
      const [oldL, old] = this.plans.entries().next().value;
      old.bufs.forEach((b) => b.destroy());
      this.plans.delete(oldL);
    }
    const p = this.buildPlan(L);
    this.plans.set(L, p);
    return p;
  }

  // Writes one window's inputs and submits its calls and the copy of its audio into the next readback buffer. Calls setUtterance first.
  // x is the window's latents [512][L] (a Float32Array, written here), or {lat, T, s, n}: frames [s, s + n) of lat, the acoustic [512][T]
  // latents (stride T), which the window's first dispatch gathers into its input (no upload). The caller keeps lat valid until this window
  // has run: the queue runs the windows and the acoustic runs in submission order. har is the window's source, [22][60 L + 1]. Returns a
  // ticket for read. At most OUTSTANDING tickets are outstanding, and they are read in submission order. The queue runs writes and
  // submissions in call order, so the next window can be submitted at once: its input writes take effect after this window's commands,
  // and its audio goes to the next readback buffer of the ring.
  submit(x, har) {
    const fromGpu = x.lat !== undefined, L = fromGpu ? x.n : x.length / 512, LH = 60 * L + 1;
    if (!this.hasAffines) throw new Error('setUtterance first');
    if (!Number.isInteger(L) || L < 1 || har.length !== 22 * LH) throw new Error(`bad inputs: ${L} frames, har ${har.length}`);
    if (fromGpu && !(Number.isInteger(x.T) && Number.isInteger(x.s) && x.s >= 0 && x.s + L <= x.T)) throw new Error(`bad latent window: frames ${x.s} to ${x.s + L} of ${x.T}`);
    if (this.tickets.length === OUTSTANDING) throw new Error(`${OUTSTANDING} windows are outstanding: read the oldest before submitting another`);
    const p = this.plan(L), d = this.device, N = SAMPLES_PER_FRAME * L;
    if (!fromGpu) d.queue.writeBuffer(p.xin, 0, x);
    d.queue.writeBuffer(p.harin, 0, har);
    const enc = d.createCommandEncoder();
    enc.clearBuffer(p.harin, p.pad.offset, p.pad.size);
    const pass = enc.beginComputePass();
    if (fromGpu) {
      d.queue.writeBuffer(this.gatherU, 0, new Uint32Array([L, x.T, x.s, 0]));
      const entries = [{ binding: 0, resource: { buffer: this.gatherU } }, { binding: 1, resource: { buffer: x.lat } },
        { binding: 2, resource: { buffer: p.xin } }];
      pass.setPipeline(this.kGather.pipe);
      pass.setBindGroup(0, d.createBindGroup({ layout: this.kGather.pipe.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(Math.ceil(L / 64), 512);
    }
    for (const c of p.calls) { pass.setPipeline(c.pipe); pass.setBindGroup(0, c.bg); pass.dispatchWorkgroups(c.gx, c.gy); }
    pass.end();
    const staging = this.staging[this.seq++ % OUTSTANDING];
    enc.copyBufferToBuffer(p.audio, 0, staging, 0, 4 * N);
    d.queue.submit([enc.finish()]);
    const ticket = { L, staging, N };
    this.tickets.push(ticket);
    return ticket;
  }

  // The audio of a submitted window (SAMPLES_PER_FRAME L float32 samples). The ticket leaves the queue only when its buffer is
  // unmapped; a buffer of the ring is reused OUTSTANDING submits later, after its window has been read.
  async read(ticket) {
    if (this.tickets[0] !== ticket) throw new Error('read the windows in submission order');
    let mapped = false;
    try {
      await ticket.staging.mapAsync(GPUMapMode.READ, 0, 4 * ticket.N);
      mapped = true;
      return new Float32Array(ticket.staging.getMappedRange(0, 4 * ticket.N).slice(0));
    } finally {
      if (mapped) ticket.staging.unmap();
      this.tickets.shift();
    }
  }

  // One window's audio: { audio } (worker.js reads it so). Outstanding tickets must be read first.
  async run(x, har) {
    if (this.tickets.length !== 0) throw new Error('read the outstanding windows before run');
    return { audio: await this.read(this.submit(x, har)) };
  }

  destroy() {
    for (const q of this.plans.values()) q.bufs.forEach((b) => b.destroy());
    [...this.keep, ...this.arenaBufs].forEach((b) => b.destroy());
    this.plans.clear();
    this.keep = [];
    this.arenaBufs = [];
    this.tickets = [];
  }
}
