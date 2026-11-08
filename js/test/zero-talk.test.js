// Tests for the Node.js entry, loaded by the package's own name (which also checks the exports map). Run with: npm test
import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { loadVoice, voices } from '@labzero-ai/zero-talk';

const ID = 'labzero-ai/zero-talk-11m';
const HELLO = 'Hello.';
const DATES = 'It opened on the 20th of February in 1940, and 48 people came.';

// SHA-256 of the WAV that the website's engine writes for each text (the same WebAssembly engine, the same text).
const WAV_SHA256 = {
  [HELLO]: '9856449c20eed34260f3e7b89d77f7ccde685fb8e1dc38fc2d118e27e1d085cf',
  [DATES]: '0621cb37f00f9118d50cb2eae0d5578848b6a7de816a9d2ffbfaa8877b4487ec',
};

let voice;
before(async () => {
  voice = await loadVoice(ID);
});

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const toSamplesLength = (chunks) => chunks.reduce((n, c) => n + c.samples.length, 0);

test('lists the one voice', () => {
  assert.deepEqual([...voices], [ID]);
});

test('the CommonJS entry exposes the same API and loads the voice', async () => {
  const api = createRequire(import.meta.url)('@labzero-ai/zero-talk');
  assert.deepEqual(Object.keys(api).sort(), ['loadVoice', 'voices']);
  assert.deepEqual(api.voices, voices);
  assert.ok(Object.isFrozen(api.voices));
  const voice = await api.loadVoice(ID, { engine: 'wasm' });
  assert.equal(voice.engine, 'wasm');
  const audio = await voice.speak(HELLO);
  assert.equal(sha256(audio.toWav()), WAV_SHA256[HELLO]);
  await assert.rejects(api.loadVoice('labzero-ai/unknown'), /valid ids/);
});

test('an unknown id rejects and lists the valid ids', async () => {
  await assert.rejects(loadVoice('labzero-ai/unknown'), (error) => {
    assert.match(error.message, /labzero-ai\/unknown/);
    assert.match(error.message, /labzero-ai\/zero-talk-11m/);
    return true;
  });
});

test('an unknown engine rejects and names the choices', async () => {
  await assert.rejects(loadVoice(ID, { engine: 'cpu' }), /engine must be 'auto', 'gpu' or 'wasm'/);
});

test("engine 'gpu' rejects in Node.js and names the fix", async () => {
  await assert.rejects(loadVoice(ID, { engine: 'gpu' }), /WebGPU.*'auto' or 'wasm'/);
});

test('the voice runs on WebAssembly in Node.js', () => {
  assert.equal(voice.engine, 'wasm');
});

test('speaks "Hello." at 24000 Hz with samples', async () => {
  const audio = await voice.speak(HELLO);
  assert.ok(audio.samples instanceof Float32Array);
  assert.equal(audio.sampleRate, 24000);
  assert.ok(audio.samples.length > 0);
  assert.equal(audio.samples.length, 20100);
  assert.equal(audio.seconds, 20100 / 24000);
  assert.deepEqual(audio.warnings, []);
});

test('steering tags are dropped from the speech and named in the warnings', async () => {
  const plain = await voice.speak(HELLO);
  const tagged = await voice.speak('[M2] Hello.');
  assert.deepEqual(tagged.samples, plain.samples);
  assert.equal(tagged.warnings.length, 1);
  assert.match(tagged.warnings[0], /steering is not supported.*\[M2\]/);
});

test('a numeric date is read as a date', async () => {
  const numeric = await voice.speak('It was 1/2/2026.');
  const written = await voice.speak('It was January 2nd twenty twenty-six.');
  assert.equal(numeric.samples.length, written.samples.length);
});

test('toWav() writes a 16-bit mono 24000 Hz WAV with an AI-generated marker', async () => {
  const audio = await voice.speak(HELLO);
  const wav = audio.toWav();
  assert.ok(wav instanceof Uint8Array);
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  const text = (at, n) => Buffer.from(wav.subarray(at, at + n)).toString('latin1');
  assert.equal(text(0, 4), 'RIFF');
  assert.equal(view.getUint32(4, true), wav.length - 8);
  assert.equal(text(8, 4), 'WAVE');
  const chunks = {};
  for (let at = 12; at < wav.length;) {
    const size = view.getUint32(at + 4, true);
    chunks[text(at, 4)] = { at: at + 8, size };
    at += 8 + size + (size & 1);
  }
  assert.deepEqual(Object.keys(chunks), ['fmt ', 'LIST', 'data']);
  const fmt = chunks['fmt '].at;
  assert.equal(view.getUint16(fmt, true), 1);          // PCM
  assert.equal(view.getUint16(fmt + 2, true), 1);      // mono
  assert.equal(view.getUint32(fmt + 4, true), 24000);  // sample rate
  assert.equal(view.getUint16(fmt + 14, true), 16);    // bits
  assert.equal(chunks.data.size, audio.samples.length * 2);
  assert.equal(chunks.data.at + chunks.data.size, wav.length);
  assert.match(text(chunks.LIST.at, chunks.LIST.size), /^INFOICMT.*AI-generated speech from Zero Talk 11M \(labzero-ai\/zero-talk-11m\)/s);
});

test('save() writes .wav and .ogg files and names the fix for other names', async () => {
  const { mkdtemp, readFile, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const folder = await mkdtemp(join(tmpdir(), 'zero-talk-'));
  try {
    const audio = await voice.speak(HELLO);
    await audio.save(join(folder, 'hello.wav'));
    assert.deepEqual(new Uint8Array(await readFile(join(folder, 'hello.wav'))), audio.toWav());
    await audio.save(join(folder, 'hello.OGG'));
    const ogg = await readFile(join(folder, 'hello.OGG'));
    assert.equal(ogg.subarray(0, 4).toString('latin1'), 'OggS');
    await assert.rejects(audio.save(join(folder, 'hello.mp3')), /save\(\) writes \.wav \(WAV\) or \.ogg \(Ogg Opus\); name the file with one of those extensions/);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});

test('toWav() of "Hello." and of the dates sentence is the website engine\'s WAV, byte for byte', async () => {
  for (const text of [HELLO, DATES]) {
    const audio = await voice.speak(text);
    assert.equal(sha256(audio.toWav()), WAV_SHA256[text], `WAV of ${JSON.stringify(text)}`);
  }
});

test('toOgg() writes an Ogg stream', async () => {
  const audio = await voice.speak(HELLO);
  const ogg = await audio.toOgg();
  assert.equal(Buffer.from(ogg.subarray(0, 4)).toString('latin1'), 'OggS');
});

test('stream() chunks concatenate to the same length as speak()', async () => {
  const spoken = await voice.speak(DATES);
  const chunks = [];
  for await (const chunk of voice.stream(DATES)) chunks.push(chunk);
  assert.ok(chunks.length >= 1);
  assert.equal(toSamplesLength(chunks), spoken.samples.length);
  const joined = new Float32Array(spoken.samples.length);
  let at = 0;
  for (const chunk of chunks) {
    joined.set(chunk.samples, at);
    at += chunk.samples.length;
  }
  assert.deepEqual(joined, spoken.samples);
});

test('play() rejects in Node.js and says what to do instead', async () => {
  const audio = await voice.speak(HELLO);
  await assert.rejects(audio.play(), /browser/);
  await assert.rejects(voice.play(HELLO), /browser/);
});

test('speak() rejects a non-string', async () => {
  await assert.rejects(voice.speak(42), /text must be a string/);
});
