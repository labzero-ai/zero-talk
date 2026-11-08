// The GPU engine keeps block-quantized weights packed and dequantizes them in its shaders (webgpu/kernels.js). This checks, without a GPU,
// that the packed layout (tensors.js packed()) gives exactly the float32 values the unpacked path (f32()) uploads: the code times the
// scale, rounded once to float32. Run with: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseTensors } from '../dist/webgpu/tensors.js';

const model = parseTensors(new Uint8Array(readFileSync(new URL('../dist/model.bin', import.meta.url))));

// What a shader does for code number j (low bits first) and scale index s of a packed tensor.
const dequant = (pk, scales, j, s) => {
  const per = 32 / pk.bits;
  const code = (pk.data[Math.floor(j / per)] << (32 - pk.bits - (j % per) * pk.bits)) >> (32 - pk.bits);
  return Math.fround(code * scales[s]);
};

test('packed weights dequantize to the values of f32(), tensor by tensor', () => {
  let checked = 0;
  for (const part of ['generator', 'student']) {
    const reader = model.part(part);
    for (const name of reader.names) {
      const probe = reader.packed(name);
      if (!probe) continue;
      const taps = probe.transposed ? probe.shape[2] : 0;
      const pk = taps ? reader.packed(name, { taps }) : probe;
      const want = reader.f32(name);
      const scales = new Float32Array(pk.data.buffer, 4 * pk.words);
      for (let i = 0; i < want.length; i++) {
        // want[i] in the layout the unpacked path makes: the file's [row][col], or for a transposed tensor w[(ci * cout + co) * k + tap].
        let row, col, fileCol;
        if (taps) {
          const [, cout, k] = pk.shape;
          const tap = i % k, co = Math.floor(i / k) % cout, ci = Math.floor(i / (k * cout));
          row = co; col = ci * k + tap; fileCol = ci * k + (k - 1 - tap);
        } else {
          row = Math.floor(i / pk.len); col = i % pk.len; fileCol = col;
        }
        const got = dequant(pk, scales, col * pk.rows + row, Math.floor(fileCol / pk.block) * pk.rows + row);
        if (!Object.is(got, want[i]) && got !== want[i]) assert.fail(`${part}.${name}[${i}]: ${got}, expected ${want[i]}`);
      }
      checked++;
    }
  }
  assert.ok(checked > 100, `only ${checked} packed tensors`);
});

test('packed() is null for tensors that are not block-quantized', () => {
  const reader = model.part('generator');
  assert.equal(reader.packed('conv_post.weight'), null);
});
