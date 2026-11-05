// The API that the browser and Node.js entries share: the voice, the audio, the WAV encoder and the channel to the engine worker.
// Nothing here uses a browser or a Node.js global, so both entries run the same code.

export const VOICE_IDS = Object.freeze(['labzero-ai/zero-talk-11m']);
export const ENGINE_NAMES = Object.freeze(['auto', 'gpu', 'wasm']);
export const SAMPLE_RATE = 24000;
const FORMAT_OGG = 1;      // the engine's formats (worker.js): 1 is Ogg Opus, 2 is float32
const FORMAT_FLOAT32 = 2;

// Checks the arguments of loadVoice and returns the engine name. Errors name the valid choices.
export function checkLoad(id, options) {
  if (!VOICE_IDS.includes(id)) {
    throw new Error(`Unknown voice '${id}'. The valid ids are: ${VOICE_IDS.join(', ')}.`);
  }
  const engine = options.engine ?? 'auto';
  if (!ENGINE_NAMES.includes(engine)) {
    throw new Error(`engine must be 'auto', 'gpu' or 'wasm', not '${engine}'.`);
  }
  return engine;
}

// Requests to the engine worker. Each request has an id; the worker's chunk and restart messages go to its onChunk, its reply
// (or error) settles it. `send` posts a message and `listen` receives the worker's messages (their data).
export class Channel {
  #send;
  #pending = new Map();
  #nextId = 1;
  #failure = null;
  #busy;

  constructor({ send, listen, busy = () => {} }) {
    this.#send = send;
    this.#busy = busy;
    listen((message) => this.#receive(message));
  }

  request(message, { transfer = [], onChunk = null } = {}) {
    if (this.#failure) return Promise.reject(this.#failure);
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      if (this.#pending.size === 0) this.#busy(true);
      this.#pending.set(id, { resolve, reject, onChunk });
      this.#send({ id, ...message }, transfer);
    });
  }

  // The worker failed: every request in flight, and every later one, rejects with this error.
  fail(error) {
    this.#failure = error;
    const pending = [...this.#pending.values()];
    this.#pending.clear();
    for (const p of pending) p.reject(error);
    this.#busy(false);
  }

  #receive(message) {
    const p = message && message.id !== undefined ? this.#pending.get(message.id) : undefined;
    if (!p) return; // progress messages carry no id
    if ('chunk' in message || 'restart' in message) {
      p.onChunk?.(message);
      return;
    }
    this.#pending.delete(message.id);
    if (this.#pending.size === 0) this.#busy(false);
    if ('error' in message) p.reject(new Error(message.error));
    else p.resolve(message);
  }
}

// The metadata every WAV file carries: it says the speech is AI-generated (a LIST/INFO chunk with a comment and the software name).
const MARK_COMMENT = 'AI-generated speech from Zero Talk 11M (labzero-ai/zero-talk-11m)';
const MARK_SOFTWARE = 'Zero Talk 11M';
function infoEntry(id, text) {
  const size = text.length + 1;                         // the text and its 0 terminator
  const entry = new Uint8Array(8 + size + (size & 1));  // one more 0 byte when the size is odd
  const view = new DataView(entry.buffer);
  for (let i = 0; i < 4; i++) entry[i] = id.charCodeAt(i);
  view.setUint32(4, size, true);
  for (let i = 0; i < text.length; i++) entry[8 + i] = text.charCodeAt(i);
  return entry;
}
const INFO_ENTRIES = [infoEntry('ICMT', MARK_COMMENT), infoEntry('ISFT', MARK_SOFTWARE)];
const LIST_SIZE = 4 + INFO_ENTRIES.reduce((n, e) => n + e.length, 0);   // "INFO" and the two entries
export const WAV_HEADER_BYTES = 44 + 8 + LIST_SIZE;

// The 16-bit mono WAV of float samples, written as the engine writes its own: each sample is clamped to [-1, 1], scaled by 32767 in single
// precision, and rounded half away from zero.
export function encodeWav(samples) {
  const dataBytes = 2 * samples.length;
  const out = new Uint8Array(WAV_HEADER_BYTES + dataBytes);
  const view = new DataView(out.buffer);
  const ascii = (at, text) => { for (let i = 0; i < 4; i++) out[at + i] = text.charCodeAt(i); };
  ascii(0, 'RIFF');
  view.setUint32(4, 36 + 8 + LIST_SIZE + dataBytes, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true);            // fmt chunk size
  view.setUint16(20, 1, true);             // PCM
  view.setUint16(22, 1, true);             // mono
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, 2 * SAMPLE_RATE, true);
  view.setUint16(32, 2, true);             // bytes per frame
  view.setUint16(34, 16, true);            // bits per sample
  ascii(36, 'LIST');
  view.setUint32(40, LIST_SIZE, true);
  ascii(44, 'INFO');
  let at = 48;
  for (const entry of INFO_ENTRIES) { out.set(entry, at); at += entry.length; }
  ascii(at, 'data');
  view.setUint32(at + 4, dataBytes, true);
  at += 8;
  for (let i = 0; i < samples.length; i++) {
    const scaled = Math.fround(Math.max(-1, Math.min(1, samples[i])) * 32767);
    view.setInt16(at + 2 * i, Math.sign(scaled) * Math.round(Math.abs(scaled)), true);
  }
  return out;
}

// The engine's info text is one line per fact; the warnings are the lines that start with "warning: ".
const WARNING = 'warning: ';
function warningsOf(info) {
  return String(info ?? '').split('\n').filter((line) => line.startsWith(WARNING)).map((line) => line.slice(WARNING.length));
}

// The samples in a reply's bytes. The engine's float32 output is exact, so the view is taken on the copy the worker made.
function floatsOf(bytes) {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

// A speech result. `play` is the entry's player (browser) or a function that rejects (Node.js).
class SpeechAudio {
  samples;
  sampleRate = SAMPLE_RATE;
  seconds;
  warnings;
  #channel;
  #play;

  constructor(samples, { warnings, channel, play }) {
    this.samples = samples;
    this.seconds = samples.length / SAMPLE_RATE;
    this.warnings = warnings;
    this.#channel = channel;
    this.#play = play;
  }

  toWav() {
    return encodeWav(this.samples);
  }

  async toOgg() {
    const copy = this.samples.slice();
    const reply = await this.#channel.request({ op: 'encode', samples: copy, format: FORMAT_OGG }, { transfer: [copy.buffer] });
    return reply.out;
  }

  async play() {
    await this.#play([{ samples: this.samples }]);
  }
}

// A loaded voice: speak, stream and play. Each call is one request to the engine worker, answered in the order it was made.
class SpeechVoice {
  engine;
  #channel;
  #requested;
  #play;

  constructor(channel, { engine, requested, play }) {
    this.engine = engine;
    this.#channel = channel;
    this.#requested = requested;
    this.#play = play;
  }

  async speak(text) {
    const reply = await this.#channel.request({ op: 'speak', text: checkText(text), format: FORMAT_FLOAT32, engine: this.#requested });
    return new SpeechAudio(floatsOf(reply.out), { warnings: warningsOf(reply.info), channel: this.#channel, play: this.#play });
  }

  // Yields the audio as the engine makes it. On WebAssembly the whole text comes as one chunk. If the GPU engine fails part-way, a
  // { restart: true } chunk comes first and the text starts again on WebAssembly.
  async *stream(text) {
    const queue = [];
    let wake = null;
    let finished = false;
    let failure = null;
    let chunksSinceRestart = 0;
    const notify = () => {
      const resume = wake;
      wake = null;
      resume?.();
    };
    this.#channel.request({ op: 'speak', text: checkText(text), format: FORMAT_FLOAT32, engine: this.#requested, chunks: true }, {
      onChunk: (message) => {
        if ('restart' in message) {
          queue.push({ samples: new Float32Array(0), restart: true });
          chunksSinceRestart = 0;
        } else {
          queue.push({ samples: message.chunk });
          chunksSinceRestart++;
        }
        notify();
      },
    }).then((reply) => {
      if (chunksSinceRestart === 0) queue.push({ samples: floatsOf(reply.out) });
      finished = true;
      notify();
    }, (error) => {
      failure = error;
      finished = true;
      notify();
    });
    for (;;) {
      if (queue.length > 0) {
        yield queue.shift();
      } else if (finished) {
        if (failure) throw failure;
        return;
      } else {
        await new Promise((resolve) => { wake = resolve; });
      }
    }
  }

  async play(text) {
    await this.#play(this.stream(text));
  }
}

function checkText(text) {
  if (typeof text !== 'string') throw new TypeError(`text must be a string, not ${typeof text}.`);
  return text;
}

// The voice an entry returns from loadVoice. `loaded` is the worker's reply to the load request.
export function createVoice(channel, { loaded, requested, play }) {
  return new SpeechVoice(channel, { engine: loaded.engine, requested, play });
}
