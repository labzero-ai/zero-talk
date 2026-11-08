// The WebGPU acoustic pre-pass runs a sentence in chunks of CHUNK columns, so its activations are the size of a chunk and only the sentence's
// latents, F0 and energy grow with the sentence (latents.js). This checks, without a GPU (a recording stand-in for the device), that the buffers
// are bounded that way, that a sentence is cut into the chunks the kernels' state carrying needs, and that the readback goes through two small
// buffers. The arithmetic itself (sample-for-sample the same as the whole-sentence form) is checked on a GPU. Run with: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseTensors } from '../dist/webgpu/tensors.js';
import * as L from '../dist/webgpu/latents.js';

globalThis.GPUMapMode ??= { READ: 1 };
const model = parseTensors(new Uint8Array(readFileSync(new URL('../dist/model.bin', import.meta.url))), { verify: false });

// A device that records what is made, written and submitted. A pass is its list of dispatches; a submission is its passes and copies.
const fakeDevice = () => {
  const dev = {
    live: new Set(), writes: [], submits: [], maps: [],
    limits: { maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456 },
    createBuffer(d) {
      const b = { size: d.size, usage: d.usage, destroy: () => dev.live.delete(b),
        mapAsync: async (mode, offset, size) => { dev.maps.push({ buffer: b, size }); },
        getMappedRange: (offset, size) => new ArrayBuffer(size), unmap() {} };
      dev.live.add(b);
      return b;
    },
    pushErrorScope() {},
    popErrorScope: async () => null,
    createBindGroup: (d) => d,
    createShaderModule: () => ({}),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createCommandEncoder() {
      const sub = { passes: [], copies: [] };
      return {
        beginComputePass() {
          const pass = [];
          sub.passes.push(pass);
          return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups: (x, y) => pass.push([x, y]), end() {} };
        },
        copyBufferToBuffer: (src, so, dst, d, n) => sub.copies.push({ src, so, dst, do: d, n }),
        finish: () => sub,
      };
    },
    queue: {
      writeBuffer: (b, offset, data) => dev.writes.push({ b, offset, data }),
      submit: (list) => dev.submits.push(...list),
      onSubmittedWorkDone: async () => {},
    },
  };
  return dev;
};
const total = (dev) => [...dev.live].reduce((a, b) => a + b.size, 0);
const READBACK = 0x1 | 0x8;

test('the chunk arena is the same for every sentence; a longer sentence adds only its latents, F0 and energy', async () => {
  const dev = fakeDevice();
  const a = await L.GpuAcoustic.create(dev, model.part('student'), { maxFrames: 600 });
  const atLoad = total(dev);
  const before = new Set(dev.live);
  await a.ensureCapacity(2600);
  assert.equal(a.maxFrames, 2600);
  const added = [...dev.live].filter((b) => !before.has(b)), removed = [...before].filter((b) => !dev.live.has(b));
  // Only the sentence's buffers were remade: the latents [512][T], the F0 and the energy, 4 * 514 bytes a frame.
  assert.deepEqual(added.map((b) => b.size).sort((x, y) => x - y), [4 * 2600, 4 * 2600, 4 * 512 * 2600]);
  assert.deepEqual(removed.map((b) => b.size).sort((x, y) => x - y), [4 * 600, 4 * 600, 4 * 512 * 600]);
  assert.equal(total(dev), atLoad + L.sentenceBytes(2600) - L.sentenceBytes(600));
  // No buffer is as large as the whole-sentence arena's activations were (1152 words a frame for 2600 frames), and the growth is a tenth of what it was.
  assert.ok(Math.max(...[...dev.live].map((b) => b.size)) <= 4 * 512 * 2600);
  const wholeSentence = 4 * 5531 * (2600 - 600);   // what the growth took when the activations were as long as the sentence (5,531 words a frame)
  assert.ok(total(dev) - atLoad < wholeSentence / 10, `${total(dev) - atLoad} bytes added by a sentence of 2600 frames`);
  // The memory the worker reports adds up to the buffers this made (the weights and the small fixed buffers are the rest).
  const m = a.memory();
  assert.equal(m.arena, L.chunkBytes);
  assert.equal(m.sentence, L.sentenceBytes(2600));
  assert.equal(m.readback, 2 * L.RING);
  assert.ok(total(dev) >= m.weights + m.arena + m.state + m.sentence + m.readback);
  assert.equal(L.CHUNK % 256, 0, 'the energy scan carries between pieces of 256 columns');
  assert.ok(L.CHUNK >= L.MAX_TOKENS, 'the text encoder runs in one piece in the chunk arena');
  a.destroy();
  assert.equal(dev.live.size, 0, 'destroy frees every buffer');
});

for (const T of [1, 300, 512, 513, 1300, 2600]) {
  test(`a run of ${T} frames is ${Math.ceil(T / L.CHUNK)} chunk(s) with the state carried`, async () => {
    const dev = fakeDevice();
    const a = await L.GpuAcoustic.create(dev, model.part('student'), { maxFrames: 600 });
    await a.ensureCapacity(T);
    dev.writes.length = 0;
    dev.submits.length = 0;
    const out = a.run(new Float32Array(384 * T), new Float32Array(T), new Float32Array(T), { read: true });
    const chunks = Math.ceil(T / L.CHUNK);
    assert.equal(dev.submits.length, chunks, 'one submission per chunk');
    // The chunk uniform before each submission: the chunk's first column (a multiple of 256) and the sentence's columns.
    const cu = dev.writes.filter((w) => w.b === a.cu).map((w) => [...w.data]);
    assert.deepEqual(cu, Array.from({ length: chunks }, (_, k) => [k * L.CHUNK, T, 0, 0]));
    dev.submits.forEach((s, k) => {
      const tc = Math.min(L.CHUNK, T - k * L.CHUNK), p = a.plan(tc);
      assert.equal(s.passes.length, 1);
      assert.equal(s.passes[0].length, p.calls.length, 'a chunk is the plan of its own length');
      for (const [x, y] of s.passes[0]) assert.ok(x >= 1 && y >= 1 && x <= 65535 && y <= 65535, 'a dispatch is the size of a chunk');
    });
    // The readback: the latents [512][T] in groups of RING bytes, the first two copied by the run, the rest as read() goes.
    const bytes = 4 * 512 * T, groups = Math.ceil(bytes / L.RING);
    assert.equal(out.readBytes, bytes);
    const first = dev.submits[chunks - 1].copies;
    assert.equal(first.length, Math.min(2, groups));
    assert.ok(dev.submits.slice(0, -1).every((s) => s.copies.length === 0));
    dev.submits.length = 0;
    const { latents } = await a.read(out);
    assert.equal(latents.length, 512 * T);
    const copies = [...first, ...dev.submits.flatMap((s) => s.copies)];
    assert.equal(copies.length, groups);
    let at = 0;
    copies.forEach((c, k) => {
      assert.equal(c.src, out.latents);
      assert.equal(c.so, at);
      assert.equal(c.dst, a.rings[k & 1], 'the two readback buffers alternate');
      assert.equal(c.n, Math.min(L.RING, bytes - at));
      at += c.n;
    });
    assert.equal(at, bytes);
    assert.ok(dev.live.size > 0 && [...dev.live].filter((b) => b.usage === READBACK).length === 4, 'two readback buffers, the durations and the frame pass');
    assert.throws(() => a.run(null, new Float32Array(T), new Float32Array(T)), /frames\(/);
    a.destroy();
  });
}

test('a frame pass is cut into the same chunks and reads back F0 and energy by column', async () => {
  const dev = fakeDevice();
  const a = await L.GpuAcoustic.create(dev, model.part('student'), { maxFrames: 2600 });
  a.textFor = 3;
  const counts40 = new Uint32Array([500, 450, 350]), F = 2 * 1300;
  dev.writes.length = 0;
  dev.submits.length = 0;
  const { f0, energy } = await a.frames(counts40, F);
  assert.equal(f0.length, F);
  assert.equal(energy.length, F);
  const chunks = Math.ceil(F / L.CHUNK);
  assert.equal(dev.submits.length, chunks);
  assert.deepEqual(dev.writes.filter((w) => w.b === a.cu).map((w) => [...w.data]), Array.from({ length: chunks }, (_, k) => [k * L.CHUNK, F, 3, 0]));
  dev.submits.forEach((s, k) => {
    const tc = Math.min(L.CHUNK, F - k * L.CHUNK), g0 = k * L.CHUNK;
    assert.deepEqual(s.copies.map((c) => [c.so, c.dst, c.do, c.n]),
      [[0, a.framesRead, 4 * g0, 4 * tc], [4 * tc, a.framesRead, 4 * (F + g0), 4 * tc]]);
  });
  assert.equal(a.framesFor, F);
  assert.throws(() => a.run(null, new Float32Array(F + 1), new Float32Array(F + 1)), /bad inputs/);
  a.destroy();
});

test('the capacities are buckets up to the limit, and a sentence above it is refused', async () => {
  const dev = fakeDevice();
  const a = await L.GpuAcoustic.create(dev, model.part('student'), { maxFrames: 600 });
  await assert.rejects(() => a.ensureCapacity(L.MAX_FRAMES + 1), /above the acoustic pre-pass's limit/);
  assert.equal(a.capacityFor(601), 800);
  assert.equal(a.capacityFor(2600), 2600);
  a.destroy();
});

test('no dispatch of a chunk binds one buffer twice (the shared arena buffers are never read and written by one dispatch)', async () => {
  const dev = fakeDevice();
  const a = await L.GpuAcoustic.create(dev, model.part('student'), { maxFrames: 600 });
  const plans = [a.plan(L.CHUNK), a.plan(7), a.plan(1), a.framesPlan(L.CHUNK), a.framesPlan(300), a.textPlan(L.MAX_TOKENS), a.textPlan(3)];
  for (const p of plans) {
    assert.ok(p.calls.length > 0);
    for (const c of p.calls) {
      const used = c.bg.entries.map((e) => e.resource.buffer);
      assert.equal(new Set(used).size, used.length, `${c.label} binds a buffer twice`);
    }
  }
  a.destroy();
});
