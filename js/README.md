# Zero Talk for JavaScript

`@labzero-ai/zero-talk`: Zero Talk 11M, an English text-to-speech model with about 11M parameters, for browsers and
Node.js 22 or later. The model ships inside the package (about 9 MB), so speech is made on your device: no speech
server or API key. Node.js reads the model from the installed package and uses no network. A browser downloads the three
model files from wherever the package is served when the voice first loads (host them yourself with `baseUrl`), and
sends nothing.

Model page and live demo: https://labzero.ai/zero-talk-11m/

## Install

```sh
npm install @labzero-ai/zero-talk
```

This package uses ES modules. In Node.js, use a `.mjs` file or set `"type": "module"` in your `package.json`.
CommonJS works from a `.cjs` file (or any `.js` file in a project without `"type": "module"`):

```js
const { loadVoice } = require('@labzero-ai/zero-talk');

(async () => {
  const voice = await loadVoice('labzero-ai/zero-talk-11m');
  const audio = await voice.speak('Hello from a small voice.');
  await audio.save('hello.wav');
})();
```

## Use

```js
import { loadVoice } from '@labzero-ai/zero-talk';

const voice = await loadVoice('labzero-ai/zero-talk-11m');
const audio = await voice.speak('Hello from a small voice.');

await audio.save('hello.wav');   // Node.js: the extension picks the format, .wav or .ogg (Ogg Opus)

audio.samples      // Float32Array, mono
audio.sampleRate   // 24000
audio.seconds
audio.toWav()      // Uint8Array, a WAV file (synchronous)
await audio.toOgg()  // Uint8Array, Ogg Opus (smaller; a promise)
await audio.play()   // browser only
```

In a browser there is no file system: `save()` rejects there. Use `toWav()` or `await audio.toOgg()` and offer the bytes
as a download.

Play while it is still being made, or take it piece by piece:

```js
await voice.play('This starts playing before the sentence is finished.');   // browser only

const parts = [];
for await (const chunk of voice.stream('A longer text, read piece by piece.')) {
  if (chunk.restart) parts.length = 0;   // the GPU engine failed: the text starts again on WebAssembly
  else parts.push(chunk.samples);        // Float32Array
}
```

In Node.js the engine is WebAssembly, which makes the whole text first: `stream()` yields one chunk there, and the audio is
the same as `speak()` returns. In a browser with WebGPU, chunks arrive while the text is still being made.

## API at a glance

| Call | Returns |
| --- | --- |
| `loadVoice(id, { engine?, baseUrl? })` | `Promise<Voice>`. `id` is `'labzero-ai/zero-talk-11m'`, the only voice in this version; `voices` lists them. |
| `voice.speak(text)` | `Promise<Audio>`: `samples`, `sampleRate`, `seconds`, `warnings`, `save(path)` (Node.js), `toWav()`, `toOgg()`, `play()` (browser) |
| `voice.stream(text)` | async iterable of `{ samples, restart? }` chunks |
| `voice.play(text)` | `Promise<void>`, resolves when playback ends (browser) |
| `voice.engine` | `'gpu'` or `'wasm'`, the engine in use |

Text: English, plain sentences; numbers, dates and prices written as digits are read out. Other scripts are skipped with a
warning on `audio.warnings`.

If `chunk.restart` is true, the GPU engine failed and speech is starting again on WebAssembly. Discard earlier chunks;
the following chunks replace them. `voice.play()` handles this automatically.

## Where the model files come from

The ID names files that ship inside the package: the engine (`engine.wasm`), the model (`model.bin`) and the text front
end (`frontend.bin`). Nothing is fetched from anywhere else.

- **With Vite**: nothing to configure. The files are referenced with `new URL('…', import.meta.url)`, so Vite copies them
  into your build and your own server serves them. The browser downloads them once, on the first `loadVoice()`, and caches
  them. webpack 5 understands the same pattern but is untested.
- **With esbuild or Rollup**: serve the package's complete `dist/` folder and import `dist/zero-talk.js` directly, as in
  the example below, or configure your build to copy its worker modules and model files. `baseUrl` changes where model
  files load from; worker modules still load beside the JavaScript entry.
- **Node.js**: read from `node_modules`, no network. Runs on the CPU (WebAssembly) in a worker thread.
- **Without a bundler**, in any web page:

  ```html
  <button id="speak" disabled>Speak</button>
  <script type="module">
    import { loadVoice } from 'https://cdn.jsdelivr.net/npm/@labzero-ai/zero-talk@1/dist/zero-talk.js';
    const voice = await loadVoice('labzero-ai/zero-talk-11m');
    const button = document.querySelector('#speak');
    button.onclick = () => voice.play('Hello.');
    button.disabled = false;
  </script>
  ```

  jsDelivr serves the published package unchanged. To avoid any third party, copy the package's `dist/` folder to your
  own server and import it from there.
- **From your own folder or CDN**: `loadVoice('labzero-ai/zero-talk-11m', { baseUrl: 'https://example.com/zero-talk/' })`
  reads the same three files and `files.json` from there.

Every file's size and SHA-256 are in the package's `model.json`, the same as on the model page.

## Engines

`engine` is `'auto'` (the default), `'gpu'` or `'wasm'`. `auto` uses WebGPU where the browser offers it on a secure page,
and WebAssembly otherwise or if the GPU fails. Node.js uses WebAssembly.

## What it does and does not do

English only, one voice, 24 kHz mono. Questions come out flat, and long sentences can lose their rhythm; short, plain
sentences work best. The full list, with what is not possible, is on the model page.

WAV and Ogg files from `toWav()` and `toOgg()` carry an AI-generated marker in their metadata; the samples are unchanged.

The audio from Node.js (WebAssembly) differs from the Python and Rust packages' audio by floating-point rounding: the same speech, not the same bytes.

Tested in Chromium on an Apple-silicon laptop and in Safari on an iPhone 17 Pro. In Node.js (22 and 24 on Linux and
Windows, and on macOS) it writes byte-for-byte the same audio as the website's engine. Other browsers and phones are
untested.

## Errors and limits

Measured on 2026-11-06 with Node.js 26.12.1 (the WebAssembly engine). A failed call rejects with an `Error` whose message
is shown; quoted texts are exact.

| Case | What happens |
| --- | --- |
| Empty text, spaces only, `...`, an emoji, or text with no letter the model can read (Chinese, for example) | `speak()` and `stream()` reject with `nothing to say: the text has no words the model can read; pass a sentence made of letters, for example "Hello."` |
| A lone `.`, `?` or `!` | Speaks as 0.29 s of audio, with no warning. |
| Letters outside a to z (Cyrillic, Greek, Chinese) in text that has other words | Left out, and named in `audio.warnings`, for example `["1 word(s) could not be read and are left out: мир"]`. Text whose only readable parts are punctuation (`Привет, мир.`) speaks 0.46 s with that warning; `Привет мир` with no punctuation rejects as above. |
| Most symbols and emoji (`#`, `~`, `/`, `😀`) | Left out with no warning. Some symbols are read as words: `&` as "and", `%` as "percent", `+` as "plus", `=` as "equals", `@` as "at", `©` as "copyright". |
| A whole number of more than 306 digits | Left out with no warning; a 307-digit number leaves only its full stop. |
| Length | A text of 5,016 characters (96 sentences) speaks in full: 306 s of speech, made in 164 s. A text of 50,000 characters (8,908 words) also speaks in full: 3,049 s of speech, made in 1,609 s, peak memory about 1.18 GB. A sentence of 2,815 characters with no full stop speaks in full: 161 s of speech, made in 91 s. The text is split into pieces of at most 510 phonemes; a longer sentence is split too, with no warning. |
| Unknown voice ID | `Unknown voice 'labzero-ai/unknown'. The valid ids are: labzero-ai/zero-talk-11m.` |
| Unknown `engine` | `engine must be 'auto', 'gpu' or 'wasm', not 'cpu'.` With `engine: 'gpu'` in Node.js: `loadVoice: engine 'gpu' needs WebGPU, which only browsers have. Use engine 'auto' or 'wasm' in Node.js.` |
| `baseUrl` folder without the files | Node.js's own error, for example `ENOENT: no such file or directory, open '<folder>/engine.wasm'`. |
| Text that is not a string | `TypeError: text must be a string, not number.` |
| `save()` with another extension (`.mp3`, none) | `<path>: save() writes .wav (WAV) or .ogg (Ogg Opus); name the file with one of those extensions`. Nothing is written. `.WAV` in capitals works. |
| `save()` into a missing folder | Node.js's own error: `ENOENT: no such file or directory, open '<path>'`. |
| `play()` in Node.js | `play() plays through a browser and does not work in Node.js. Write the audio to a file with audio.save("hello.wav").` |

Streaming: in Node.js `stream()` yields one chunk, the whole text, made before the chunk is returned; for a three-sentence
text that is 1 chunk of 90,900 samples, the same as `speak()`.

Calls can overlap: several `speak()` calls at once give the same audio as one after another.

## Licence

Apache-2.0, for the code and the model files. Zero Talk builds on Kokoro-82M (Apache-2.0). Third-party notices are in
[THIRD-PARTY-NOTICES.txt](https://github.com/labzero-ai/zero-talk/blob/main/js/dist/THIRD-PARTY-NOTICES.txt).

Lab Zero · https://labzero.ai
