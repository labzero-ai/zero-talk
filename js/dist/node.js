// Zero Talk 11M in Node.js. The engine runs on the CPU (WebAssembly) in a worker thread. The model files are read from the package folder
// (or from baseUrl: a folder path, a file: URL or an http(s) URL). Nothing is fetched unless baseUrl is an http(s) URL.
import { Worker } from 'node:worker_threads';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { Channel, checkLoad, createVoice, VOICE_IDS } from './voice.js';

export const voices = VOICE_IDS;

/**
 * Loads a voice. Resolves with a voice once the engine and the files are ready.
 * @param {string} id 'labzero-ai/zero-talk-11m'
 * @param {{ engine?: 'auto' | 'gpu' | 'wasm', baseUrl?: string | URL }} [options]
 * @returns {Promise<import('../index.d.ts').Voice>}
 */
export async function loadVoice(id, options = {}) {
  const engine = checkLoad(id, options);
  if (engine === 'gpu') {
    throw new Error("loadVoice: engine 'gpu' needs WebGPU, which only browsers have. Use engine 'auto' or 'wasm' in Node.js.");
  }
  const files = await readFiles(options.baseUrl);
  const worker = new Worker(new URL('./node-worker.js', import.meta.url));
  // The worker keeps the process alive only while a request is in flight, so a script ends when its work is done.
  const channel = new Channel({
    send: (message, transfer) => worker.postMessage(message, transfer),
    listen: (receive) => worker.on('message', receive),
    busy: (active) => (active ? worker.ref() : worker.unref()),
  });
  worker.on('error', (error) => channel.fail(error));
  worker.on('exit', () => channel.fail(new Error('the engine worker stopped')));
  try {
    const transfer = ['engine.wasm', 'model.bin', 'frontend.bin'].map((name) => files[name]);
    const loaded = await channel.request({ op: 'load', engine: 'wasm', files }, { transfer });
    return createVoice(channel, { loaded, requested: 'wasm', play: playInNode, save: saveInNode });
  } catch (error) {
    await worker.terminate();
    throw new Error(`loadVoice: ${error instanceof Error ? error.message : String(error)}`);
  }
}

async function saveInNode(path, bytes) {
  try {
    await writeFile(path, bytes);
  } catch (error) {
    if (error?.code === 'ENOENT') throw new Error(`${path}: No such file or directory; create the folder first`, { cause: error });
    throw error;
  }
}

async function playInNode() {
  throw new Error('play() plays through a browser and does not work in Node.js. Write the audio to a file with audio.save("hello.wav").');
}

// The three model files, and files.json (their sizes), read as bytes.
const FILE_NAMES = ['engine.wasm', 'model.bin', 'frontend.bin', 'files.json'];
async function readFiles(baseUrl) {
  const addresses = baseUrl == null
    ? {
        'engine.wasm': new URL('./engine.wasm', import.meta.url),
        'model.bin': new URL('./model.bin', import.meta.url),
        'frontend.bin': new URL('./frontend.bin', import.meta.url),
        'files.json': new URL('./files.json', import.meta.url),
      }
    : Object.fromEntries(FILE_NAMES.map((name) => [name, new URL(name, folderUrl(baseUrl))]));
  const files = {};
  for (const [name, url] of Object.entries(addresses)) {
    const bytes = await readBytes(url, name);
    files[name] = name.endsWith('.json') ? JSON.parse(new TextDecoder().decode(bytes)) : bytes;
  }
  return files;
}

// baseUrl as a folder URL: a path (relative to the working folder), a file: URL or an http(s) URL.
function folderUrl(baseUrl) {
  const text = String(baseUrl);
  const url = /^[a-z][a-z0-9+.-]*:/i.test(text) ? new URL(text) : pathToFileURL(resolve(text));
  if (!url.pathname.endsWith('/')) url.pathname += '/';
  return url;
}

// The bytes at a file: or http(s) URL, as an ArrayBuffer (a copy that can be transferred to the worker).
async function readBytes(url, name) {
  if (url.protocol === 'file:') {
    const bytes = await readFile(fileURLToPath(url));
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status} from ${url.origin}`);
  return response.arrayBuffer();
}
