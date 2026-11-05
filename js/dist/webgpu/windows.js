// The window geometry of the WebGPU generator's runs, used by the worker. A sentence of F frames (80 Hz, SAMPLES_PER_FRAME output samples each) is cut into cores; a run covers its core
// plus HALO frames on each side, clipped to the sentence, and keeps the core's samples. The first core is short, so the first
// audio comes early. SAMPLES_PER_FRAME must agree with the WebAssembly engine.
export const SAMPLES_PER_FRAME = 300;
export const FIRST_CORE = 16;
export const CORE = 60;
export const HALO = 12;

// Frames [s, e) of a sentence: latents xs[:, s:e] (512 x F) and source columns har[:, 60 s : 60 e + 1] (22 x (60 F + 1)).
export function windowOf(xs, har, F, s, e) {
  const n = e - s, LH = 60 * F + 1, Ln = 60 * n + 1;
  const wx = new Float32Array(512 * n), wh = new Float32Array(22 * Ln);
  for (let r = 0; r < 512; r++) wx.set(xs.subarray(r * F + s, r * F + e), r * n);
  for (let r = 0; r < 22; r++) wh.set(har.subarray(r * LH + 60 * s, r * LH + 60 * e + 1), r * Ln);
  return { x: wx, har: wh };
}

// The cores of a sentence of F frames whose first kept sample is trim (output samples): each {c0, c1, s, e} runs frames [s, e)
// and keeps the samples of frames [c0, c1). The first core starts at the frame that holds trim and is firstCore long, the others
// core long (FIRST_CORE and CORE by default; the halo is HALO on each side).
export function cores(F, trim, firstCore = FIRST_CORE, core = CORE) {
  const out = [];
  let c0 = Math.floor(trim / SAMPLES_PER_FRAME), size = firstCore;
  while (c0 < F) {
    const c1 = Math.min(F, c0 + size);
    out.push({ c0, c1, s: Math.max(0, c0 - HALO), e: Math.min(F, c1 + HALO) });
    size = core;
    c0 = c1;
  }
  return out;
}
