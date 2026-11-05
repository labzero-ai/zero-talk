// Per-utterance AdaIN affines of the generator, as in the reference implementation. Takes the reader that parseTensors returns.

/** Length coordinates per voice row: `voice.coefficients` is [rows][COORDS]. */
export const COORDS = 4;

// The order of the AdaIN sites: each ResBlock has three units (dilations 1, 3, 5), and
// each unit has adain1.u then adain2.u.
const BLOCKS = ['noise_res.0', 'resblocks.0', 'resblocks.1', 'resblocks.2',
  'noise_res.1', 'resblocks.3', 'resblocks.4', 'resblocks.5'];
const UNITS = 3;
const PREFIX = 'generator.';
const FOLD = '.fold';

/** The four length coordinates of voice row `voiceIndex` (a copy). */
export function lengthCoords(model, voiceIndex) {
  const row = Number(voiceIndex);
  const [rows, cols] = model.shape('voice.coefficients');
  if (cols !== COORDS || !Number.isInteger(row) || row < 0 || row >= rows) {
    throw new Error(`voice row ${voiceIndex} is not in voice.coefficients [${rows}][${cols}]`);
  }
  return model.f32('voice.coefficients').slice(row * COORDS, (row + 1) * COORDS);
}

/** The generator's AdaIN sites (the names without `generator.` and `.fold`), in the order of BLOCKS. */
export function siteNames(model) {
  const sites = [];
  for (const block of BLOCKS) {
    for (let u = 0; u < UNITS; u++) sites.push(`${block}.adain1.${u}`, `${block}.adain2.${u}`);
  }
  const folds = model.names.filter((n) => n.startsWith(PREFIX) && n.endsWith(FOLD));
  if (folds.length !== sites.length || sites.some((s) => !model.has(PREFIX + s + FOLD))) {
    throw new Error(`the generator's fold tensors do not match the ${sites.length} expected AdaIN sites`);
  }
  return sites;
}

/**
 * Per-site affines for one utterance. `moments` maps each site to {mean, variance} (a Map or a plain object).
 * With (gamma, beta) = fold[0] + sum_r coords[r] * fold[1 + r] (float64), scale[i] = (1 + gamma[i]) * w[i] /
 * sqrt(variance[i] + eps) and shift[i] = (1 + gamma[i]) * b[i] + beta[i]. Returns Map site -> {mean, scale, shift}.
 */
export function siteAffines(model, coords, moments) {
  const eps = parseFloat(model.meta.norm_eps);
  const affines = new Map();
  for (const site of siteNames(model)) {
    const m = moments instanceof Map ? moments.get(site) : moments[site];
    if (!m) throw new Error(`no moments for ${site}`);
    const fold = model.f32(`${PREFIX}${site}${FOLD}`);
    const w = model.f32(`${PREFIX}${site}.norm.weight`);
    const b = model.f32(`${PREFIX}${site}.norm.bias`);
    const c = w.length;
    const n = 2 * c;
    if (fold.length !== (1 + COORDS) * n || b.length !== c || m.mean.length !== c || m.variance.length !== c) {
      throw new Error(`${site}: fold, norm and moment sizes do not match`);
    }
    // The sum starts at -0.0; that differs from 0.0 only
    // in the sign of an exact zero.
    const h = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = -0;
      for (let r = 0; r < COORDS; r++) s += coords[r] * fold[(1 + r) * n + i];
      h[i] = fold[i] + s;
    }
    const scale = new Float32Array(c);
    const shift = new Float32Array(c);
    for (let i = 0; i < c; i++) {
      const g = 1 + h[i];
      scale[i] = (g * w[i]) / Math.sqrt(m.variance[i] + eps);
      shift[i] = g * b[i] + h[c + i];
    }
    affines.set(site, { mean: Float32Array.from(m.mean), scale, shift });
  }
  return affines;
}
