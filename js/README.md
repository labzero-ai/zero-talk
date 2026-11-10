# Zero Talk for JavaScript

`@labzero-ai/zero-talk`: Zero Talk 11M, an English text-to-speech model with about 11M parameters, for browsers and
Node.js 22 or later. The model ships inside the package (the model file is 7.05 MB; with the engine and the text front end, about 9 MB), so speech is made on your device: no speech server,
no API key. Made for reading short English texts aloud where size and privacy matter. Not for long narration, other
languages or imitating real people.

Model page and live demo: https://labzero.ai/zero-talk-11m/

## Install and first audio

```sh
npm install @labzero-ai/zero-talk
```

Save this as `speak.mjs` and run `node speak.mjs`:

```js
import { loadVoice } from '@labzero-ai/zero-talk';

const voice = await loadVoice('labzero-ai/zero-talk-11m');
const audio = await voice.speak('Hello from a small voice.');
await audio.save('hello.ogg');   // Ogg Opus, 24 kHz mono: 1.8 s of speech in 5 KB
```

Name the file `hello.wav` for an uncompressed WAV (17 times larger). In a CommonJS file (`.cjs`, or `.js` in a project
without `"type": "module"`), use `const { loadVoice } = require('@labzero-ai/zero-talk');` inside an `async` function.

## Use

```js
audio.samples        // Float32Array, mono, 24,000 samples per second, values from -1 to 1
audio.sampleRate     // 24000
audio.seconds
await audio.encode('wav')                    // Uint8Array, a WAV file
await audio.encode('ogg', { bitrate: 64 })   // Uint8Array, Ogg Opus at 64 kbit/s (8 to 128; 24 when omitted)
await audio.play()   // browser only
```

In a browser there is no file system: `save()` rejects there. Use `await audio.encode('ogg')` and offer the bytes as
a download (a `Blob`).

Play while it is still being made, or take it piece by piece:

```js
await voice.play('This starts playing before the sentence is finished.');   // browser only

const parts = [];
for await (const chunk of voice.stream('A longer text, read piece by piece.')) {
  if (chunk.restart) parts.length = 0;   // the GPU engine failed: the text starts again on WebAssembly
  else parts.push(chunk.samples);        // Float32Array, 24,000 samples per second
}
```

In Node.js the engine is WebAssembly, which makes the whole text first: `stream()` yields one chunk there, and the audio is the
same as `speak()` returns. In a browser with WebGPU, chunks arrive while the text is still being made. To save a stream's
audio as a file, call `speak()` and `save()`: the audio is the same.

## API at a glance

| Call | Returns |
| --- | --- |
| `loadVoice(id, { engine?, modelDir?, files? })` | `Promise<Voice>`. `id` is `'labzero-ai/zero-talk-11m'`, the only voice in this version; `voices` lists them. `modelDir` and `files` say where the model files are (see below). |
| `voice.speak(text)` | `Promise<Audio>`: `samples`, `sampleRate`, `seconds`, `warnings`, `save(path)` (Node.js), `encode(format, { bitrate? })` (a promise of bytes; `format` is `'wav'` or `'ogg'`), `play()` (browser) |
| `voice.stream(text)` | async iterable of `{ samples, restart? }` chunks |
| `voice.play(text)` | `Promise<void>`, resolves when playback ends (browser) |
| `voice.engine` | `'gpu'` or `'wasm'`, the engine in use |

If `chunk.restart` is true, the GPU engine failed and speech is starting again on WebAssembly. Discard earlier chunks;
the following chunks replace them. `voice.play()` handles this automatically.

## Text it reads

English, plain sentences. The front end rewrites written forms as the words a person would say before the voice reads them.
Examples of that rewriting step (digits are turned into words afterwards):

| Written | Rewritten as |
| --- | --- |
| `Elm Ave.` (more text follows) | `Elm Avenue` |
| `9:30 a.m.` | `9 thirty A M` |
| `800-555-0199` | digits, one by one |
| `2.75 kg`, `10 km` | `two point seven five kilograms`, `10 kilometers` |
| `18 °C` | `18 degrees Celsius` |
| `2026-11-05` | `November 5th twenty twenty-six` |
| `example.com` | `example dot com` |
| `€20` | `20 euros` |

A `St.` at the very end of a text is not expanded. Other scripts are skipped with a warning on `audio.warnings`.

## Where the model files come from

The ID names files that ship inside the package: the engine (`engine.wasm`), the model (`model.bin`) and the text front
end (`frontend.bin`). Nothing is fetched from anywhere else, and nothing is downloaded from Hugging Face.

- **With Vite**: nothing to configure. The files are referenced with `new URL('…', import.meta.url)`, so Vite copies them
  into your build and your own server serves them. The browser fetches them on the first `loadVoice()`; its normal HTTP
  cache keeps them if your server sends long cache headers. webpack 5 understands the same pattern but is untested.
- **With esbuild or Rollup**: serve the package's complete `dist/` folder and import `dist/zero-talk.js` directly, as in
  the example below, or configure your build to copy its worker modules and model files. `modelDir` and `files` change where model
  files load from; worker modules still load beside the JavaScript entry.
- **Node.js**: read from `node_modules`, no network. Runs on the CPU (WebAssembly) in a worker thread.
- **Without a bundler**, in any web page:

  ```html
  <button id="speak" disabled>Speak</button>
  <script type="module">
    import { loadVoice } from 'https://cdn.jsdelivr.net/npm/@labzero-ai/zero-talk@1.0.0/dist/zero-talk.js';
    const voice = await loadVoice('labzero-ai/zero-talk-11m');
    const button = document.querySelector('#speak');
    button.onclick = () => voice.play('Hello.');
    button.disabled = false;
  </script>
  ```

  jsDelivr serves the published package unchanged, and the exact version in the link keeps it that way. To avoid any
  third party, copy the package's `dist/` folder to your own server and import it from there.
- **From your own folder or CDN**: `loadVoice('labzero-ai/zero-talk-11m', { modelDir: 'https://example.com/zero-talk/' })`
  reads `engine.wasm`, `model.bin` and `frontend.bin` from that folder (relative addresses are relative to the page; in Node.js a
  path, a `file:` URL or an `http(s)` URL).
- **Each file at its own address**, for hosting that cannot follow the folder layout (hashed file names, another origin for the
  `.wasm`): `loadVoice('labzero-ai/zero-talk-11m', { files: { engine: '/a/engine.3fa9.wasm', model: '/b/weights.bin', frontend: '/b/frontend.bin' } })`.
  A file named in `files` wins over `modelDir`, and `modelDir` over the package's own files. A server on another origin must send
  an `Access-Control-Allow-Origin` header; if it does not, the error says so and names the address.

Every file's size and SHA-256 are in the package's `model.json`, the same as on the model page.

## Privacy

No text you give it and no audio it makes leaves the device. Node.js makes no network request. A browser requests only the
package's own files (about 9 MB), from whichever host serves them: your server, or jsDelivr if you use the link above.

## Speed and memory

Measured with Zero Talk 1.0 on 2026-11-09 (UTC), reading the 503-character text in
[`bench/text.txt`](https://github.com/labzero-ai/zero-talk/blob/main/bench/text.txt) (32.8 s of speech) with Node.js 26 on an
Apple M1 Pro (10 cores, 32 GB): 1.5 times real time (seconds of speech made per second of compute), peak memory about
215 MB, voice load 0.2 s. Memory grows with the text: a 5,030-character text (328 s of speech) ran at 1.6 times real time and peaked at about 230 MB, and 50,000 characters peaked at about 1.2 GB (measured 2026-11-06). In a browser with WebGPU the speed
depends on the GPU, and the GPU buffers take about 34 to 37 MB; the demo on the model page shows the speed on your device.
`bench/node.mjs` in the repository measures your own machine.

## Why this one

About 9 MB with the engine, against 68 to 341 MB for the files Pocket TTS and Kokoro-82M install (not counting PyTorch).
On 52 everyday sentences read back by a speech recogniser, the word error rate was 6.5% to 6.7% (95% intervals 4.1 to 9.6%),
level with Pocket TTS 18.2M and Kokoro-82M, whose intervals overlap (trial of 2026-11-06). It has one voice and reads
English only. More detail is in the [repository README](https://github.com/labzero-ai/zero-talk#why-this-one).

## What it does and does not do

English only, one voice, 24 kHz mono. Questions come out flat and exclamations do not come across; long sentences can lose
their rhythm; some S sounds have a slight lisp; "Hi, …" greetings can sound unnatural; some names are read oddly; there are
no tempo or pause controls. Short, plain sentences work best. The full list is on the model page.

Silent cases: most symbols and emoji are left out without a warning, whole numbers of more than 306 digits are dropped,
and a lone `.` speaks as 0.29 s of audio.

WAV and Ogg files from `encode()` and `save()` carry an AI-generated label in their metadata: a comment in both
and, in Ogg, the IPTC digital source type `trainedAlgorithmicMedia`. It is a label, not a watermark: re-encoding or stripping
metadata removes it, and `audio.samples` carries none.

The audio from Node.js (WebAssembly) differs from the Python and Rust packages' audio by floating-point rounding: the same
speech, not the same bytes.

Tested in Chromium on an Apple-silicon laptop and in Safari on an iPhone 17 Pro. In Node.js (22 and 24 on Linux and
Windows, and on macOS) it writes byte-for-byte the same audio as the website's engine. Other browsers and phones are
untested.

## FAQ

**Can it clone a voice or imitate a person?** No. It has one fixed voice and no way to pass a reference recording.

**Why Ogg Opus and not MP3?** Ogg Opus is an open codec with small files at this quality (24 kbit/s). Some old players cannot
open it; convert once with `ffmpeg -i hello.ogg hello.mp3`.

**Does it send my text anywhere?** No. See Privacy above.

**Where do I report a problem?** [GitHub issues](https://github.com/labzero-ai/zero-talk/issues).

## Errors and limits

Measured on 2026-11-06 with Node.js 26.12.1 (the WebAssembly engine). A failed call rejects with an `Error` whose message
is shown; quoted texts are exact.

| Case | What happens |
| --- | --- |
| Empty text, spaces only, `...`, an emoji, or text with no letter the model can read (Chinese, for example) | `speak()` and `stream()` reject with `nothing to say: the text has no words the model can read; pass a sentence made of letters, for example "Hello."` |
| Steering tags (`[M2]`, `[breathy:1]`) | Not supported: removed before reading, and named in `audio.warnings`. Links `[label](target)` and `[aside]` are read as written. |
| A lone `.`, `?` or `!` | Speaks as 0.29 s of audio, with no warning. |
| Letters outside a to z (Cyrillic, Greek, Chinese) in text that has other words | Left out, and named in `audio.warnings`, for example `["1 word(s) could not be read and are left out: мир"]`. Text whose only readable parts are punctuation (`Привет, мир.`) speaks 0.46 s with that warning; `Привет мир` with no punctuation rejects as above. |
| Most symbols and emoji (`#`, `~`, `/`, `😀`) | Left out with no warning. Some symbols are read as words: `&` as "and", `%` as "percent", `+` as "plus", `=` as "equals", `@` as "at", `©` as "copyright". |
| A whole number of more than 306 digits | Left out with no warning; a 307-digit number leaves only its full stop. |
| Length | A text of 5,016 characters (96 sentences) and a text of 50,000 characters (8,908 words) speak in full, not truncated; peak memory about 1.2 GB for 50,000 characters. A sentence of 2,815 characters with no full stop also speaks in full. The text is split into pieces of at most 510 phonemes (the units of sound the model reads); a longer sentence is split too, with no warning. |
| GPU memory (`engine: 'gpu'`) | About 34–37 MB of GPU buffers for any text length. On a slow GPU the engine switches to a larger window after two slow requests and then uses about 43–46 MB. |
| Unknown voice ID | `Unknown voice 'labzero-ai/unknown'. The valid ids are: labzero-ai/zero-talk-11m.` |
| Unknown `engine` | `engine must be 'auto', 'gpu' or 'wasm', not 'cpu'.` With `engine: 'gpu'` in Node.js: `loadVoice: engine 'gpu' needs WebGPU, which only browsers have. Use engine 'auto' or 'wasm' in Node.js.` |
| `modelDir` folder without the files (Node.js) | `engine.wasm not found at <folder>/engine.wasm; modelDir must be a folder that holds engine.wasm, model.bin and frontend.bin, or files must name each file`. In a browser a missing file is `<url>: HTTP 404; modelDir must be …`. |
| Unknown option or file key | `Unknown option 'baseUrl'. The options are: engine, modelDir, files.`; `Unknown file 'wasm' in files. The names are: engine, model, frontend.` |
| Text that is not a string | `TypeError: text must be a string, not number.` |
| `save()` with another extension (`.mp3`, none) | `<path>: save() writes .wav (WAV) or .ogg (Ogg Opus); name the file with one of those extensions`. Nothing is written. `.WAV` in capitals works. |
| `encode()` with another format, a bit rate outside 8 to 128, or a bit rate for `'wav'` | `format must be 'wav' or 'ogg', not 'mp3'.`; `bitrate must be a whole number of kbit/s from 8 to 128, not 999.`; `bitrate is for 'ogg'; 'wav' has no bit rate.`; an unknown option: `encode: unknown option 'kbps'; the only option is bitrate (kbit/s, for 'ogg').` |
| `save()` into a missing folder | `<path>: No such file or directory; create the folder first` |
| `play()` in Node.js | `play() plays through a browser and does not work in Node.js. Write the audio to a file with audio.save("hello.ogg").` |

Streaming: in Node.js `stream()` yields one chunk, the whole text, made before the chunk is returned; for a three-sentence
text that is 1 chunk of 90,900 samples, the same as `speak()`.

Calls can overlap: several `speak()` calls at once give the same audio as one after another.

## Licence

Apache-2.0, for the code and the model files. Zero Talk builds on Kokoro-82M (Apache-2.0). Third-party notices are in
[THIRD-PARTY-NOTICES.txt](https://github.com/labzero-ai/zero-talk/blob/main/js/dist/THIRD-PARTY-NOTICES.txt).

Lab Zero · https://labzero.ai
