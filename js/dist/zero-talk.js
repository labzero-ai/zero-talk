// Zero Talk 11M in a browser. Import this module from the package with a bundler, or with a script tag from a CDN: it starts the engine
// in a module worker and reads the three model files beside this module (or from baseUrl). Nothing is sent over the network at run time.
import { Channel, checkLoad, createVoice, VOICE_IDS } from './voice.js';
import { playStream } from './play.js';

export const voices = VOICE_IDS;

/**
 * Loads a voice. Resolves with a voice once the engine and the files are ready.
 * @param {string} id 'labzero-ai/zero-talk-11m'
 * @param {{ engine?: 'auto' | 'gpu' | 'wasm', baseUrl?: string | URL }} [options]
 * @returns {Promise<import('../index.d.ts').Voice>}
 */
export async function loadVoice(id, options = {}) {
  const engine = checkLoad(id, options);
  const urls = fileUrls(options.baseUrl);
  const { worker, blobUrl } = startWorker();
  const channel = new Channel({
    send: (message, transfer) => worker.postMessage(message, transfer),
    listen: (receive) => { worker.onmessage = (event) => receive(event.data); },
  });
  worker.onerror = (event) => channel.fail(new Error(event.message || 'the engine worker stopped'));
  try {
    const loaded = await channel.request({ op: 'load', engine, urls });
    return createVoice(channel, { loaded, requested: engine, play: playStream });
  } catch (error) {
    worker.terminate();
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(engine === 'gpu'
      ? `loadVoice: engine 'gpu' is not available (${reason}). Use engine 'auto' or 'wasm'.`
      : `loadVoice: ${reason}`);
  } finally {
    if (blobUrl) URL.revokeObjectURL(blobUrl);
  }
}

// The files' addresses: beside this module by default, or in baseUrl (a folder, with or without its trailing slash).
const FILE_NAMES = ['engine.wasm', 'model.bin', 'frontend.bin', 'files.json'];
function fileUrls(baseUrl) {
  if (baseUrl == null) {
    return {
      'engine.wasm': new URL('./engine.wasm', import.meta.url).href,
      'model.bin': new URL('./model.bin', import.meta.url).href,
      'frontend.bin': new URL('./frontend.bin', import.meta.url).href,
      'files.json': new URL('./files.json', import.meta.url).href,
    };
  }
  const base = new URL(baseUrl, location.href);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return Object.fromEntries(FILE_NAMES.map((name) => [name, new URL(name, base).href]));
}

// The engine worker. A module worker must come from the page's own origin. When this module is served from another origin (a CDN), the
// worker is started from a same-origin blob that imports the worker module, which then loads its own files from the CDN.
function startWorker() {
  const sameOrigin = new URL('./worker.js', import.meta.url).origin === location.origin;
  if (sameOrigin) {
    return { worker: new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }), blobUrl: null };
  }
  const source = `import ${JSON.stringify(new URL('./worker.js', import.meta.url).href)};`;
  const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  return { worker: new Worker(blobUrl, { type: 'module' }), blobUrl };
}
