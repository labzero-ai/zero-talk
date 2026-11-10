// Every JavaScript file the package ships must parse. The Node.js tests never load the browser entry or the GPU modules, so a syntax
// error there would otherwise ship unseen.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const modules = readdirSync(dist, { recursive: true }).filter((name) => /\.(js|cjs)$/.test(name));

test('the package ships JavaScript files', () => {
  assert.ok(modules.includes('zero-talk.js') && modules.includes('webgpu/generator.js'), modules.join(', '));
});

for (const name of modules) {
  test(`${name} parses`, () => {
    execFileSync(process.execPath, ['--check', `${dist}${name}`], { stdio: 'pipe' });
  });
}
