// The engine, off the page's thread: a browser worker, or the worker_threads worker of Node.js (node-worker.js gives it the same
// message port). It instantiates engine.wasm from the bytes it is given (a browser fetches them from URLs; Node.js passes bytes read
// from disk) and loads the model and the front end. Where WebGPU works, the generator runs on the GPU (the modules in webgpu/), and so does the acoustic pre-pass that makes the latents. The text encoder, the durations and the frame pass run on the GPU too,
// while the rounding to 40 Hz counts (Bresenham), the plan, the source and the moments run in WebAssembly. Otherwise, or when the GPU
// fails, the whole pipeline runs in WebAssembly. A text's own error (no phonemes, too many tokens) is a reply error that leaves the GPU
// engine in use.
// Requests: {id, op: 'load', engine?, urls? | files?}. engine is 'auto' (the default), 'wasm' or 'gpu' (which fails instead of falling
// back). urls maps each file name to an absolute URL to fetch; files maps each name to its bytes, and 'files.json' to its parsed sizes.
// {id, op: 'speak', text, format?, engine?, chunks?}. format 0 is WAV (the default), 1 is Ogg Opus, 2 is float32. chunks true asks for
// {id, chunk, sentence} messages before the reply, one per window as the GPU makes it, and {id, restart: true} when a GPU speak failed and
// is made again in WebAssembly (the consumer then drops the chunks it has).
// {id, op: 'encode', samples, format}. The samples (a Float32Array) encoded by the engine's encoder as WAV, Ogg Opus or float32.
// Replies: {id, ...result} or {id, error}. While the files load: {progress: {bytes, total}}, with no id.
const SEED = 1729; // the engine's default seed
const ENGINES = ['auto', 'wasm', 'gpu'];
// The adaptation's threshold (verdict): a request whose median window time over its core's audio is above SLOW (0.25, four times real
// time) is slow, and two slow requests in a row move the next request to the 120-frame core.
const SLOW = 0.25;
const ACOUSTIC_FRAMES = 600; // the acoustic pre-pass's arena at the start (latents.js's first bucket); a longer sentence grows it

// The GPU modules are imported statically (not on demand) so bundlers can pack this worker as one file with their default
// settings; they only define code, and the WebGPU calls happen in startGpu.
import { GpuGenerator, OUTSTANDING } from './webgpu/generator.js';
import { parseTensors } from './webgpu/tensors.js';
import * as condition from './webgpu/condition.js';
import * as windows from './webgpu/windows.js';
import * as acousticModule from './webgpu/latents.js';

const enc = new TextEncoder(), dec = new TextDecoder();
let x = null;                  // the wasm instance's exports
let modelBytes = null, frontendBytes = null; // kept from the load: zt_load and zt_load_front read them again when needed
let full = false;              // zt_load (the whole WebAssembly pipeline) has run in this instance
let gen = null, gpuDevice = null, gpuModel = null, gpuAffines = null; // the GPU engine: its generator and device, the model reader, siteAffines
let acoustic = null, acousticLib = null; // the acoustic pre-pass (its GpuAcoustic and the module's helpers); null when it is not in use
let acousticStatus = 'off';      // the acoustic pre-pass for the replies: 'on', 'off' 'failed:<reason>' (it could not be made)
let acousticLimit = 0;           // the most frames a sentence may have on the acoustic pre-pass: the smaller of MAX_FRAMES and the device's
let win = null, outstanding = 0; // the window geometry (windows.js) and how many windows the generator keeps in flight
let gpuFailure = null;         // why the GPU engine stopped (the first reason); the calls after that use WebAssembly
let adaptOff = false;          // the adaptation is off for the session (a growth failed)
let adaptFailure = null;       // why the adaptation stopped (the replies' adaptFailure)
let coreFrames = 0;            // the GPU core of the next request: windows.js's CORE, or 120 once the adaptation has moved to it
let slowStreak = 0;            // requests in a row that produced a verdict and were slow
let siteCache = null;          // the site names and their channel counts: the same for every sentence of the model

const view = (ptr, len) => new Uint8Array(x.memory.buffer, ptr, len);
const str = (ptr, len) => dec.decode(view(ptr, len));
const lastError = () => str(x.zt_error_ptr(), x.zt_error_len());
const info = () => str(x.zt_info_ptr(), x.zt_info_len());
// Typed arrays copied out of the module's memory (a grown memory detaches the views taken before the growth).
const f32At = (ptr, n) => new Float32Array(x.memory.buffer, ptr, n).slice();
const u32At = (ptr, n) => new Uint32Array(x.memory.buffer, ptr, n).slice();
// The source's har of a sentence of F frames (22 rows of 60 F + 1 columns), a view of the module's memory: take it again after a call that may grow the memory.
const harOf = (F) => new Float32Array(x.memory.buffer, x.zt_gpu_har_ptr(), 22 * (60 * F + 1));
// An error of the text's front half (no phonemes, too many tokens, a bad sentence): the text's fault, so speak rethrows it and the GPU
// engine stays in use.
const frontError = (msg) => Object.assign(new Error(msg), { front: true });

// Copies bytes into the module's memory; free them with zt_free after the call.
function put(bytes) {
  const ptr = x.zt_alloc(bytes.length);
  view(ptr, bytes.length).set(bytes);
  return ptr;
}

// Copies float32 samples into the module's memory at a 4-byte boundary (zt_alloc aligns nothing); free with zt_free(base, bytes).
function putF32(samples) {
  const bytes = 4 * samples.length + 4, base = x.zt_alloc(bytes), ptr = (base + 3) & ~3;
  new Float32Array(x.memory.buffer, ptr, samples.length).set(samples);
  return { base, ptr, bytes };
}
// fn(...pointers) with each array copied into the module's memory (putF32), freed after the call; fn's result is returned.
function withF32(arrays, fn) {
  const blocks = arrays.map((a) => putF32(a));
  try {
    return fn(...blocks.map((b) => b.ptr));
  } finally {
    for (const b of blocks) x.zt_free(b.base, b.bytes);
  }
}

// The body of a file, as one array. The size from files.json is allocated first and each part is copied in as it arrives, so the bytes
// are copied once. A size that is missing or wrong makes the array grow (and it is cut to the bytes read): the same bytes either way.
async function fetchCounted(url, size, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  let out = new Uint8Array(Number.isInteger(size) && size > 0 ? size : 0), at = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (at + value.length > out.length) {
      const grown = new Uint8Array(Math.max(2 * out.length, at + value.length));
      grown.set(out.subarray(0, at));
      out = grown;
    }
    out.set(value, at);
    at += value.length;
    onBytes(value.length);
  }
  return at === out.length ? out : out.slice(0, at);
}

// zt_load (the whole pipeline) or zt_load_front (the front end, the student and the conditioner, not the generator's weights).
function callLoad(name) {
  const pm = put(modelBytes), pf = put(frontendBytes);
  const rc = x[name](pm, modelBytes.length, pf, frontendBytes.length);
  const error = rc ? lastError() : '';
  x.zt_free(pm, modelBytes.length);
  x.zt_free(pf, frontendBytes.length);
  if (rc) throw new Error(error);
}

// The student that zt_load_front leaves out, loaded the first time a sentence runs its front half in WebAssembly (the GPU path's fallback).
let studentLoaded = false;
function ensureStudent() {
  if (studentLoaded) return;
  const pm = put(modelBytes);
  const rc = x.zt_load_student(pm, modelBytes.length);
  const error = rc ? lastError() : '';
  x.zt_free(pm, modelBytes.length);
  if (rc) throw new Error(error);
  studentLoaded = true;
}

function ensureFull() {
  if (!full) {
    callLoad('zt_load');
    full = true;
  }
}

// Stops the GPU engine for good: the first reason is kept, and the generator's and the acoustic pre-pass's buffers and the device are destroyed.
function stopGpu(reason) {
  gpuFailure ??= reason;
  gen?.destroy();
  gen = null;
  acoustic?.destroy();
  acoustic = null;
  gpuDevice?.destroy();
  gpuDevice = null;
}

// The acoustic pre-pass (webgpu/latents.js) on the device: the model's student part, its weights and an arena for ACOUSTIC_FRAMES frames, made in error
// scopes like the generator's. Throws with the reason when it is not made.
async function makeAcoustic(device, model) {
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  let oom = null, invalid = null, b = null;
  try {
    b = await acousticLib.GpuAcoustic.create(device, model.part('student'), { maxFrames: ACOUSTIC_FRAMES });
  } finally {
    oom = await device.popErrorScope();
    invalid = await device.popErrorScope();
  }
  if (oom || invalid) {
    b?.destroy();
    throw new Error(`the acoustic was not made (${oom ? `out of memory: ${oom.message}` : `validation: ${invalid.message}`})`);
  }
  return b;
}

// The GPU engine: a device, the generator's weights and pipelines on it, the acoustic pre-pass (when asked for and made), and the front half of the
// pipeline in WebAssembly. It throws, with the reason, when a step fails, and frees what it made. The GPU modules are imported here only,
// because they use WebGPU's globals when they load. The modules and the model are read before the device is requested, so a failure there
// leaks no device. An acoustic pre-pass that is not made does not fail the engine: it runs without the acoustic pre-pass, and the reason is kept for the replies.
async function startGpu(useAcoustic) {
  if (!navigator.gpu) throw new Error('WebGPU is not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  acousticLib = acousticModule;
  // The model's checksum is not read here: zt_load_front checks the same bytes (which parses them), before the GPU engine runs anything.
  const model = parseTensors(modelBytes, { verify: false });
  const device = await adapter.requestDevice();
  gpuDevice = device;
  device.lost.then((lost) => { if (lost.reason !== 'destroyed') stopGpu(`device lost (${lost.reason})`); });
  device.addEventListener('uncapturederror', (e) => stopGpu(String(e.error?.message ?? e.error)));
  const core = windows.CORE;
  let g = null, b = null, status = 'off';
  try {
    // The generator's allocations run in error scopes: an out-of-memory or validation error is reported here, at the start.
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    let oom = null, invalid = null;
    try {
      g = await GpuGenerator.create(device, model, { maxFrames: core + 2 * windows.HALO });
    } finally {
      oom = await device.popErrorScope();
      invalid = await device.popErrorScope();
    }
    if (oom || invalid) throw new Error(`the GPU generator was not made (${oom ? `out of memory: ${oom.message}` : `validation: ${invalid.message}`})`);
    if (gpuFailure) throw new Error(`the GPU engine stopped while it was made: ${gpuFailure}`);
    callLoad('zt_load_front');
    studentLoaded = false;
    if (useAcoustic) {
      try {
        b = await makeAcoustic(device, model);
        status = 'on';
      } catch (e) {
        status = `failed:${String((e && e.message) || e)}`;
      }
    }
    if (gpuFailure) throw new Error(`the GPU engine stopped while the acoustic was made: ${gpuFailure}`);
  } catch (e) {
    g?.destroy();
    b?.destroy();
    gpuDevice = null;
    device.destroy();
    throw e;
  }
  gen = g;
  acoustic = b;
  acousticStatus = status;
  acousticLimit = b ? Math.min(acousticLib.MAX_FRAMES, acousticLib.frameLimit(device)) : 0;
  gpuModel = model;
  gpuAffines = condition.siteAffines;
  win = windows;
  outstanding = OUTSTANDING;
  coreFrames = core;
}

// The files, the instance, and the engine: the GPU engine when it starts (engine 'auto' or 'gpu'), else the WebAssembly pipeline.
async function load({ engine = 'auto', urls = null, files = null }) {
  if (!ENGINES.includes(engine)) throw new Error(`unknown engine '${engine}'`);
  const sizes = files ? files['files.json'] : await fetchJson(urls['files.json']);
  const total = sizes.total ?? 0;
  let bytes = 0;
  const onBytes = (n) => {
    bytes += n;
    self.postMessage({ progress: { bytes, total } });
  };
  const names = ['engine.wasm', 'model.bin', 'frontend.bin'];
  const [wasm, model, frontend] = await Promise.all(names.map((name) => (files
    ? new Uint8Array(files[name])
    : fetchCounted(urls[name], sizes.files?.[name], onBytes))));
  x = (await WebAssembly.instantiate(wasm, {})).instance.exports;
  modelBytes = model;
  frontendBytes = frontend;
  let fallbackReason = null;
  acousticStatus = 'off';
  if (engine !== 'wasm') {
    try {
      await startGpu(true);
    } catch (e) {
      if (engine === 'gpu') throw e;
      fallbackReason = String((e && e.message) || e);
    }
  }
  if (!gen) ensureFull();
  return { bytes: wasm.byteLength + model.byteLength + frontend.byteLength, engine: gen ? 'gpu' : 'wasm', fallbackReason,
    acoustic: gen ? acousticStatus : 'off', acousticMemory: acoustic ? acoustic.memory() : null };
}

// A small JSON file (files.json), from its URL.
async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

// The WebAssembly engine: the whole text in one call (zt_speak).
function wasmSpeak(text, format) {
  ensureFull();
  const bytes = enc.encode(text), ptr = put(bytes);
  const t0 = performance.now();
  const rc = x.zt_speak(ptr, bytes.length, SEED, format);
  const ms = performance.now() - t0;
  x.zt_free(ptr, bytes.length);
  if (rc) throw new Error(lastError());
  return { out: view(x.zt_out_ptr(), x.zt_out_len()).slice(), seconds: x.zt_duration(), ms, info: info(), engine: 'wasm', firstChunkMs: null, chunks: 0 };
}

// The site names (without ".norm") and the channels of each site, as the prepared sentence gives them.
function siteInfo() {
  const names = str(x.zt_gpu_site_names_ptr(), x.zt_gpu_site_names_len()).split('\n');
  const count = x.zt_gpu_site_count();
  if (names.length !== count) throw frontError(`${names.length} site names for ${count} sites`);
  return { names, channels: u32At(x.zt_gpu_site_channels_ptr(), count) };
}

// The conditioner's moments of the sentence as Map site -> {mean, variance}, read from the flat list (for each site in the order of
// the names: the mean, then the variance, each of as many values as the site has channels). The names and the channels are the same for
// every sentence of the model, so they are read at the first sentence.
function momentsOf() {
  siteCache ??= siteInfo();
  const { names, channels } = siteCache;
  const flat = f32At(x.zt_gpu_moments_ptr(), x.zt_gpu_moments_len());
  const moments = new Map();
  let at = 0;
  names.forEach((name, k) => {
    const C = channels[k];
    moments.set(name, { mean: flat.subarray(at, at + C), variance: flat.subarray(at + C, at + 2 * C) });
    at += 2 * C;
  });
  if (at !== flat.length) throw frontError(`${flat.length} moment values for ${at} expected`);
  return moments;
}

// The soft limiter (zt_post_limit) on a copy of the samples, as speak_into applies it to each push.
function limit(samples) {
  const m = putF32(samples);
  x.zt_post_limit(m.ptr, samples.length);
  const out = f32At(m.ptr, samples.length);
  x.zt_free(m.base, m.bytes);
  return out;
}

// The samples as the engine encodes them (zt_encode: WAV, Ogg Opus or float32); the bytes are copied out.
function encode(samples, format) {
  const m = putF32(samples);
  const rc = x.zt_encode(m.ptr, samples.length, format);
  const error = rc ? lastError() : '';
  x.zt_free(m.base, m.bytes);
  if (rc) throw new Error(error);
  return { out: view(x.zt_out_ptr(), x.zt_out_len()).slice(), seconds: x.zt_duration() };
}

function concat(parts) {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

// The GPU core of the next request, decided at the start of a speak (never during one): after two slow requests in a row, the core becomes
// 120 frames and the generator's arena grows to its windows (it never shrinks back). A growth that fails keeps the 60-frame core, turns the
// adaptation off for the session and leaves the engine on the GPU; only a failed restore of the arena is thrown (speak stops the GPU
// engine). "In a row" counts requests that produced a verdict: a request without one (fewer than two samples) leaves the streak as it is.
async function adaptCore() {
  if (adaptOff || coreFrames >= 120 || slowStreak < 2) return;
  try {
    await gen.ensureCapacity(120 + 2 * win.HALO);
  } catch (e) {
    if (e?.fatal) throw e;
    adaptOff = true;
    adaptFailure = String((e && e.message) || e);
    return;
  }
  coreFrames = 120;
}

// The verdict on a request's generator speed: the median of its full interior windows' times (at least two samples; else no verdict).
// A median above SLOW makes the request slow, and the streak grows; a request with a verdict that is not slow resets it.
function verdict(samples) {
  if (samples.length < 2) return;
  const s = [...samples].sort((a, b) => a - b), mid = s.length >> 1;
  const median = s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  slowStreak = median > SLOW ? slowStreak + 1 : 0;
}

// The GPU engine's speak. Per sentence, prep gives the moments' affines, the voice's
// coordinates and the latents: from the acoustic pre-pass (acousticSentence: its front half, on the GPU or in WebAssembly, then the pre-pass), or from
// WebAssembly (localSentence, which makes the whole spectrum). On
// the acoustic pre-pass the source's spectrum is made per window: the first window's while the pre-pass runs, the rest as the windows are submitted
// (submit), overlapping the GPU's earlier windows. The generator runs the sentence's windows (cores with their halos, windows.js). The
// first kept sample starts the first window; the samples before the trim are dropped as speak_into drops them. Each piece is limited in
// WebAssembly and sent as it is made. At most `outstanding` windows are in flight, read in submission order. The acoustic pre-pass has one latents
// buffer, and sentence i+1's prep (its front half and pre-pass) runs only after the loop has read all of sentence i's windows and made
// their source (prep runs after the drain), so no window reads a latents buffer or a source that a later run has overwritten.
async function gpuSpeak(text, format, send) {
  const t0 = performance.now();
  const bytes = enc.encode(text), ptr = put(bytes);
  const n = x.zt_gpu_begin(ptr, bytes.length);
  x.zt_free(ptr, bytes.length);
  if (n < 0) throw frontError(lastError());
  const sentences = info();
  const core = coreFrames, onAcoustic = acoustic !== null;
  let lastRead = 0; // when the last window's audio was read (performance.now)
  const r = { frontMs: performance.now() - t0, gpuWaitMs: 0, windows: 0, parts: [], firstChunkMs: null, chunks: 0,
    prepSources: [], frontSources: [], frontFallbacks: [], acousticFallbacks: [], times: [] };
  const inFlight = [];
  const samples = []; // this request's full-interior-window times, over their core's audio (clamped to 1)

  // Sentence i's front half on the GPU : zt_gpu_prepare (the token ids and the coordinates), the text
  // encoder on the GPU (acoustic.encode: the durations), zt_gpu_counts (the 40 Hz counts, the frames and the trim), the arena grown, the frame
  // pass on the GPU (acoustic.frames: the raw F0 and energy) and zt_gpu_plan. Its frames stay in the acoustic pre-pass's XB (frames null). Null when the
  // acoustic pre-pass does not take the sentence. A failed step throws (acousticFront falls back for the sentence); the text's own error is a frontError.
  const frontOnGpu = async (i, tm) => {
    const tg = performance.now();
    try {
      let tw = performance.now();
      if (x.zt_gpu_prepare(i, SEED) !== 0) throw frontError(lastError());
      const tokens = x.zt_gpu_ids_len(), ids = u32At(x.zt_gpu_ids_ptr(), tokens), coords = f32At(x.zt_gpu_coords_ptr(), 4);
      tm.wasmFrontMs += performance.now() - tw;
      const pending = acoustic.encode(acousticLib.textInput(acoustic.tables, ids, coords), tokens);
      const dur = await pending;
      tw = performance.now();
      if (withF32([dur], (p) => x.zt_gpu_counts(p, tokens)) !== 0) throw new Error(lastError());
      const F = x.zt_gpu_frames(), trim = x.zt_gpu_trim(), counts40 = u32At(x.zt_gpu_counts_ptr(), tokens);
      tm.wasmFrontMs += performance.now() - tw;
      if (F > acousticLimit) return notTaken(i, F);
      await grow(i, F, tm);
      const framesPending = acoustic.frames(counts40, F);
      const { f0: rawF0, energy: rawE } = await framesPending;
      tw = performance.now();
      if (withF32([rawF0, rawE], (p, q) => x.zt_gpu_plan(p, q, F)) !== 0) throw new Error(lastError());
      const front = { F, trim, coords, f0: f32At(x.zt_gpu_front_f0_ptr(), F), energy: f32At(x.zt_gpu_front_energy_ptr(), F), frames: null,
        source: 'gpu' };
      tm.wasmFrontMs += performance.now() - tw;
      return front;
    } finally {
      tm.gpuFrontMs += performance.now() - tg;
    }
  };
  // The acoustic pre-pass does not take sentence i of F frames (above acousticLimit): the reason is recorded, and the sentence runs in WebAssembly.
  const notTaken = (i, F) => {
    r.acousticFallbacks.push(`sentence ${i}: ${F} frames is above the acoustic's limit of ${acousticLimit}`);
    return null;
  };
  // Sentence i's front half in WebAssembly (zt_gpu_front: the text encoder, durations, frames and plan), its frames copied for the acoustic pre-pass's
  // run. Null when the acoustic pre-pass does not take the sentence.
  const frontOnWasm = async (i, tm) => {
    const tf = performance.now();
    ensureStudent();
    if (x.zt_gpu_front(i, SEED) !== 0) throw frontError(lastError());
    const F = x.zt_gpu_frames(), trim = x.zt_gpu_trim();
    if (F > acousticLimit) {
      tm.wasmFrontMs += performance.now() - tf;
      return notTaken(i, F);
    }
    const front = { F, trim, frames: f32At(x.zt_gpu_front_frames_ptr(), 384 * F), f0: f32At(x.zt_gpu_front_f0_ptr(), F),
      energy: f32At(x.zt_gpu_front_energy_ptr(), F), coords: f32At(x.zt_gpu_coords_ptr(), 4), source: 'wasm' };
    tm.wasmFrontMs += performance.now() - tf;
    await grow(i, F, tm);
    return front;
  };
  // The acoustic pre-pass's arena holds F frames (acoustic.ensureCapacity, a no-op once it does); its time is growMs. A failure is tagged capacity: it is
  // not a GPU front's failure (acousticFront rethrows it), so the sentence gets one attempt, as it did before the GPU front existed.
  const grow = async (i, F, tm) => {
    const tg = performance.now();
    try {
      await acoustic.ensureCapacity(F);
    } catch (e) {
      throw Object.assign(e, { capacity: true });
    } finally {
      tm.growMs += performance.now() - tg;
    }
  };
  // Sentence i's front half for the acoustic pre-pass: the GPU's (frontOnGpu) when it is on, else WebAssembly's (frontOnWasm). A GPU front that fails
  // non-fatally (not a text's error, not a fatal one, not an arena growth's, and the GPU engine still running) is recorded in frontFallbacks,
  // and WebAssembly fronts the sentence instead.
  const acousticFront = async (i, tm) => {
    try {
      return await frontOnGpu(i, tm);
    } catch (e) {
      if (e?.front || e?.fatal || e?.capacity || gpuFailure) throw e;
      r.frontFallbacks.push(`sentence ${i}: ${String((e && e.message) || e)}`);
    }
    return frontOnWasm(i, tm);
  };
  // Sentence i on the acoustic pre-pass: its front half (acousticFront), the pre-pass submitted (not awaited), zt_gpu_source for the first window in
  // WebAssembly while the GPU runs (submit makes the rest, window by window), the latents read back, their statistics given to zt_gpu_moments;
  // the affines. Null when the acoustic pre-pass does not take the sentence (recorded in acousticFallbacks), or on an acoustic pre-pass error that is not a device
  // failure (recorded too). A text's error and a device failure are thrown (a fatal error of the acoustic pre-pass is one: the GPU engine is then stopped).
  const acousticSentence = async (i, tm) => {
    let read = null;
    try {
      const front = await acousticFront(i, tm);
      if (front === null) return null;
      const { F, trim, coords, f0, energy, frames, source } = front;
      const tsub = performance.now();
      const run = acoustic.run(frames, f0, energy, { read: true }); // submitted, not awaited: the GPU runs while zt_gpu_source does
      read = acoustic.read(run); // the mapping follows the pre-pass on the queue; the wait after the source is what the overlap did not hide
      tm.acousticSubmitMs += performance.now() - tsub;
      const firstEnd = win.cores(F, trim, win.FIRST_CORE, core)[0]?.e ?? 0; // the first window's end frame: its source is made here
      const ts = performance.now();
      if (x.zt_gpu_source(firstEnd) !== 0) throw new Error(lastError());
      tm.sourceMs += performance.now() - ts;
      const tw = performance.now();
      const { latents } = await read;
      tm.acousticWaitMs += performance.now() - tw;
      const tmo = performance.now();
      const stats = acousticLib.latentStatistics(latents, F);
      if (!stats.every(Number.isFinite)) throw new Error('the acoustic latents are not finite');
      if (withF32([stats], (p) => x.zt_gpu_moments(p, stats.length)) !== 0) throw new Error(lastError());
      const affines = gpuAffines(gpuModel, coords, momentsOf());
      tm.momentsMs += performance.now() - tmo;
      return { F, trim, lat: run.latents, tm, affines, source }; // submit makes the rest of the source
    } catch (e) {
      if (read) await read.catch(() => {}); // the run's readback is free for the next run once its read has settled
      if (gpuFailure || e?.fatal || e?.front) throw e;
      r.acousticFallbacks.push(`sentence ${i}: ${String((e && e.message) || e)}`);
      return null;
    }
  };
  // The whole front half in WebAssembly (zt_gpu_sentence): the latents are copied out for the windows to upload.
  const localSentence = (i, tm) => {
    const tf = performance.now();
    ensureStudent();
    if (x.zt_gpu_sentence(i, SEED) !== 0) throw frontError(lastError());
    const F = x.zt_gpu_frames(), trim = x.zt_gpu_trim();
    const xs = f32At(x.zt_gpu_x_ptr(), 512 * F), har = harOf(F).slice();
    tm.wasmFrontMs += performance.now() - tf;
    const tmo = performance.now();
    const affines = gpuAffines(gpuModel, f32At(x.zt_gpu_coords_ptr(), 4), momentsOf());
    tm.momentsMs += performance.now() - tmo;
    return { F, trim, lat: null, xs, har, affines };
  };
  // Sentence i's front half: the acoustic pre-pass's when it takes the sentence, else WebAssembly's. Its times are kept per sentence.
  const prep = async (i) => {
    if (gpuFailure) throw new Error(gpuFailure);
    const t = performance.now();
    const tm = { wasmFrontMs: 0, gpuFrontMs: 0, sourceMs: 0, acousticSubmitMs: 0, acousticWaitMs: 0, growMs: 0, momentsMs: 0 };
    let S = onAcoustic ? await acousticSentence(i, tm) : null;
    S ??= localSentence(i, tm);
    r.prepSources.push(S.lat ? 'acoustic' : 'local');
    r.frontSources.push(S.lat ? S.source : 'local');
    r.times.push(tm);
    r.frontMs += performance.now() - t;
    return S;
  };
  // Submits core c of sentence i; its piece keeps the samples of its frames, and the sentence's first piece drops `drop` samples.
  // An acoustic pre-pass sentence's source is made up to
  // this window's end first (a call keeps what is made, so the first window's is a no-op), and its har is a view taken after the call (sourceWindow copies the columns out at once).
  const submit = (i, S, c, drop) => {
    let w;
    if (S.lat) {
      const ts = performance.now();
      if (x.zt_gpu_source(c.e) !== 0) throw new Error(lastError());
      S.tm.sourceMs += performance.now() - ts;
      w = { x: { lat: S.lat, T: S.F, s: c.s, n: c.e - c.s }, har: acousticLib.sourceWindow(harOf(S.F), S.F, c.s, c.e) };
    } else {
      w = win.windowOf(S.xs, S.har, S.F, c.s, c.e);
    }
    const at = performance.now();
    inFlight.push({
      ticket: gen.submit(w.x, w.har), at, sentence: i, c0: c.c0, c1: c.c1, s: c.s, drop,
      full: c.e - c.s === core + 2 * win.HALO, first: r.windows === 0,
    });
    r.windows++;
  };
  // Reads the oldest window in flight and sends its piece: the drop, then the limiter. A device error reported meanwhile stops the request
  // (the page restarts it in WebAssembly, so no window is streamed from a failed device).
  const deliver = async () => {
    const w = inFlight.shift();
    const t = performance.now();
    const audio = await gen.read(w.ticket);
    const done = performance.now();
    r.gpuWaitMs += done - t;
    if (gpuFailure) throw new Error(gpuFailure);
    // A window with a non-finite sample is a GPU failure: the request restarts in WebAssembly (the limiter would pass a NaN on).
    if (!audio.every(Number.isFinite)) throw new Error("the GPU window's audio is not finite");
    // A sample: the window's time from its submit (or the previous read, when that is later, since the windows run in order), over the
    // core's audio (12.5 ms a frame). Only a full interior window counts (both halos whole); the request's first window does not.
    if (w.full && !w.first) samples.push(Math.min(1, (done - Math.max(w.at, lastRead)) / (core * 12.5)));
    lastRead = done;
    let piece = audio.slice(win.SAMPLES_PER_FRAME * (w.c0 - w.s), win.SAMPLES_PER_FRAME * (w.c1 - w.s));
    if (w.drop) piece = piece.slice(w.drop);
    const limited = limit(piece);
    if (r.chunks === 0) r.firstChunkMs = performance.now() - t0;
    r.chunks++;
    r.parts.push(limited);
    if (send) {
      const copy = limited.slice();
      send({ chunk: copy, sentence: w.sentence }, [copy.buffer]);
    }
  };

  let S = await prep(0);
  for (let i = 0; i < n; i++) {
    gen.setUtterance(S.affines);
    const cs = win.cores(S.F, S.trim, win.FIRST_CORE, core);
    for (let k = 0; k < cs.length; k++) {
      if (inFlight.length === outstanding) await deliver();
      submit(i, S, cs[k], k === 0 ? S.trim % win.SAMPLES_PER_FRAME : 0);
      if (r.chunks === 0) await deliver(); // the request's first window is read at once: the earliest first audio
    }
    while (inFlight.length) await deliver();
    if (i + 1 < n) S = await prep(i + 1);
  }
  verdict(samples);
  const { out, seconds } = encode(concat(r.parts), format);
  const total = (k) => r.times.reduce((a, t) => a + t[k], 0), per = (k) => r.times.map((t) => t[k]);
  const allAcoustic = r.prepSources.every((s) => s === 'acoustic'), noAcoustic = r.prepSources.every((s) => s === 'local');
  const allGpu = r.frontSources.every((s) => s === 'gpu'), anyGpu = r.frontSources.some((s) => s === 'gpu');
  return {
    out, seconds, ms: performance.now() - t0, info: sentences, engine: 'gpu', firstChunkMs: r.firstChunkMs, chunks: r.chunks,
    frontMs: r.frontMs, gpuWaitMs: r.gpuWaitMs, windows: r.windows, core, adaptFailure,
    acoustic: !onAcoustic ? acousticStatus : allAcoustic ? 'used' : noAcoustic ? 'unused' : 'partial',
    gpuFront: !onAcoustic ? 'off' : allGpu ? 'used' : anyGpu ? 'partial' : 'unused',
    acousticCapacity: onAcoustic ? acoustic.maxFrames : null, prepSources: r.prepSources, acousticFallbacks: r.acousticFallbacks,
    frontSources: r.frontSources, frontFallbacks: r.frontFallbacks,
    wasmFrontMs: total('wasmFrontMs'), gpuFrontMs: total('gpuFrontMs'), sourceMs: total('sourceMs'), acousticSubmitMs: total('acousticSubmitMs'), acousticWaitMs: total('acousticWaitMs'),
    growMs: total('growMs'), momentsMs: total('momentsMs'),
    perSentence: { wasmFrontMs: per('wasmFrontMs'), gpuFrontMs: per('gpuFrontMs'), sourceMs: per('sourceMs'), acousticSubmitMs: per('acousticSubmitMs'),
      acousticWaitMs: per('acousticWaitMs'), growMs: per('growMs'), momentsMs: per('momentsMs') },
  };
}

// Speaks on the engine the request names. 'auto' uses the GPU engine while it works; when a GPU speak fails, the request is made again in
// WebAssembly (after a restart message when the request asked for chunks), and the reply says why ('gpu' fails instead of falling back).
// A text's own front-half error is rethrown as it is: the GPU engine is not at fault and stays in use. The adaptation (adaptCore) runs
// first; its fatal error is a GPU failure like any other.
async function speak({ text, format = 0, engine = 'auto', chunks = false }, send) {
  if (!ENGINES.includes(engine)) throw new Error(`unknown engine '${engine}'`);
  if (engine !== 'wasm' && gen && !gpuFailure) {
    try {
      await adaptCore();
      return await gpuSpeak(text, format, chunks ? send : null);
    } catch (e) {
      if (e?.front) throw e;
      stopGpu(String((e && e.message) || e));
      if (engine === 'gpu') throw new Error(`the GPU engine failed: ${gpuFailure}`);
      if (chunks) send({ restart: true });
    }
  }
  if (engine === 'gpu') throw new Error(gpuFailure ? `the GPU engine failed: ${gpuFailure}` : 'the GPU engine is not loaded');
  const r = wasmSpeak(text, format);
  return gpuFailure ? { ...r, fallbackReason: gpuFailure } : r;
}

// One request at a time: the instance and the generator are not re-entered.
let queue = Promise.resolve();
self.onmessage = ({ data: m }) => {
  queue = queue.then(() => handle(m));
};

async function handle(m) {
  const send = (msg, transfer = []) => self.postMessage({ id: m.id, ...msg }, transfer);
  try {
    if (m.op === 'load') {
      send(await load(m));
    } else if (m.op === 'speak') {
      const r = await speak(m, send);
      send(r, [r.out.buffer]);
    } else if (m.op === 'encode') {
      const r = encode(m.samples, m.format);
      send(r, [r.out.buffer]);
    } else {
      throw new Error(`unknown request '${m.op}'`);
    }
  } catch (e) {
    send({ error: String((e && e.message) || e) });
  }
}
