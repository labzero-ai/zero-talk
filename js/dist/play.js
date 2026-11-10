// Plays speech in a browser. Each chunk is scheduled on one AudioContext right after the chunk before it, so playback starts with the
// first chunk and runs without gaps.
import { SAMPLE_RATE } from './voice.js';

let context = null;

// Plays the chunks of an async iterable as they arrive. A restart chunk drops the audio that is scheduled and not yet heard. Resolves when
// the last chunk has finished playing.
export async function playStream(chunks) {
  const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Context) throw new Error('play() needs the Web Audio API, which this browser does not have. Use speak() and encode() instead.');
  context ??= new Context({ sampleRate: SAMPLE_RATE });
  if (context.state === 'suspended') await context.resume();
  let sources = [];
  let at = context.currentTime + 0.05;
  for await (const chunk of chunks) {
    if (chunk.restart) {
      for (const source of sources) {
        try { source.stop(); } catch { /* already ended */ }
      }
      sources = [];
      at = context.currentTime + 0.05;
      continue;
    }
    const buffer = context.createBuffer(1, chunk.samples.length, SAMPLE_RATE);
    buffer.copyToChannel(chunk.samples, 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.start(at);
    sources.push(source);
    at += buffer.duration;
  }
  const remaining = at - context.currentTime;
  await new Promise((resolve) => setTimeout(resolve, Math.max(0, remaining * 1000)));
}
