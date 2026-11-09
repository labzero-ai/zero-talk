// Real-time factor of the npm package in Node.js: node bench/node.mjs [text file]
import { readFileSync } from 'node:fs';
import { loadVoice } from '../js/dist/node.js';
const text = readFileSync(process.argv[2] ?? new URL('./text.txt', import.meta.url), 'utf8').trim();
const t0 = performance.now();
const voice = await loadVoice('labzero-ai/zero-talk-11m');
const t1 = performance.now();
const audio = await voice.speak(text);
const t2 = performance.now();
console.log(JSON.stringify({ load_s: +((t1 - t0) / 1000).toFixed(2), speak_s: +((t2 - t1) / 1000).toFixed(2), audio_s: +audio.seconds.toFixed(1), rtf: +(audio.seconds / ((t2 - t1) / 1000)).toFixed(2) }));
await voice.close?.();
process.exit(0);
