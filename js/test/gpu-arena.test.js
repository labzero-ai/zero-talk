// The WebGPU generator shares its activation buffers between layers whose lifetimes do not overlap (generator.js layoutArena). This
// checks, without a GPU (a recording stand-in for the device), that the shared layout is sound for every window length: no dispatch reads
// and writes one buffer, no two values that are alive together share a buffer, every plan fits its buffers, and the arena is smaller than
// one buffer per activation. Run with: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseTensors } from '../dist/webgpu/tensors.js';
import { GpuGenerator, layoutArena } from '../dist/webgpu/generator.js';

const model = parseTensors(new Uint8Array(readFileSync(new URL('../dist/model.bin', import.meta.url))), { verify: false });
const pipe = { getBindGroupLayout: () => ({}) };
const fakeDevice = () => ({
  live: new Set(),
  createBuffer(d) { const b = { size: d.size, destroy: () => this.live.delete(b) }; this.live.add(b); return b; },
  createBindGroup: (d) => d,
  createShaderModule: () => ({}),
  createComputePipelineAsync: async () => pipe,
  queue: { writeBuffer() {} },
});

test('layoutArena shares buffers only between values that are never alive together', () => {
  const vals = [{ id: 0, name: 'a', words: 10, first: 0, last: 1 }, { id: 1, name: 'b', words: 6, first: 1, last: 2 },
    { id: 2, name: 'c', words: 8, first: 2, last: 3 }, { id: 3, name: 'd', words: 4, first: 3, last: 4 }];
  const { of, sizes } = layoutArena(vals, 5);
  assert.notEqual(of[0], of[1]);   // a is read by the dispatch that writes b
  assert.notEqual(of[1], of[2]);
  assert.equal(of[0], of[2]);      // a is dead when c is written
  assert.deepEqual([...sizes].sort((x, y) => x - y), [6, 10]);
  assert.throws(() => layoutArena([{ id: 0, name: 'x', words: 4, first: Infinity, last: -1 }], 1), /never used/);
});

for (const M of [84, 144]) {
  test(`generator arena for ${M} frames: sound sharing`, async () => {
    const dev = fakeDevice();
    const g = new GpuGenerator(dev, model, M);
    await g.ready;
    const lay = g.layout, n = lay.calls.length;
    // Lifetimes from the recorded calls (the inputs are written before the first call, the audio is copied after the last).
    const first = lay.names.map(() => Infinity), last = lay.names.map(() => -1);
    lay.calls.forEach((c, i) => [...c.reads, ...c.writes].forEach((id) => { first[id] = Math.min(first[id], i); last[id] = Math.max(last[id], i); }));
    lay.names.forEach((name, id) => { if (name === 'xin' || name === 'harin') first[id] = -1; if (name === 'audio') last[id] = n; });
    for (let a = 0; a < lay.names.length; a++) {
      assert.ok(last[a] >= first[a] && first[a] !== Infinity, `${lay.names[a]} is used`);
      for (let b = a + 1; b < lay.names.length; b++) {
        if (lay.of[a] === lay.of[b]) assert.ok(last[a] < first[b] || last[b] < first[a], `${lay.names[a]} and ${lay.names[b]} overlap in one buffer`);
      }
    }
    // A dispatch never reads and writes one buffer, nor writes one buffer twice; a buffer written in place (the stream's accumulate)
    // is bound once. The values that are read are written by an earlier call, or are inputs.
    const written = new Set(['xin', 'harin']);
    lay.calls.forEach((c, i) => {
      const w = c.writes.map((id) => lay.of[id]), r = c.reads.map((id) => lay.of[id]);
      assert.equal(new Set(w).size, w.length, `call ${i} ${c.label} writes a buffer twice`);
      for (const b of w) assert.ok(!r.includes(b), `call ${i} ${c.label} reads and writes buffer ${b}`);
      for (const id of c.reads) assert.ok(written.has(lay.names[id]), `call ${i} ${c.label} reads ${lay.names[id]} before it is written`);
      for (const id of c.writes) written.add(lay.names[id]);
    });
    // Plans of several lengths bind buffers that are large enough, and the arena is smaller than one buffer per value.
    for (const L of [1, 7, 16, 60, M]) {
      const p = g.plan(L);
      assert.equal(p.calls.length, n);
      assert.ok(p.xin.size >= 4 * 512 * L && p.harin.size >= 4 * 24 * (60 * L + 1) && p.audio.size >= 4 * 300 * L);
    }
    assert.ok(lay.words < lay.unshared / 2, `${lay.words} words against ${lay.unshared} unshared`);
    assert.ok(lay.words >= lay.peak, 'the arena holds at least what is alive at the peak');
    g.destroy();
    assert.equal(dev.live.size, 0, 'destroy frees every buffer');
  });
}
