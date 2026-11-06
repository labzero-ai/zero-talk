/**
 * Zero Talk 11M: an English text-to-speech model that runs on the device, in browsers and Node.js.
 *
 * ```js
 * import { loadVoice } from '@labzero-ai/zero-talk';
 *
 * const voice = await loadVoice('labzero-ai/zero-talk-11m');
 * const audio = await voice.speak('Hello from a small voice.');
 * audio.samples;      // Float32Array, mono, 24000 per second
 * audio.toWav();      // Uint8Array, a WAV file
 * ```
 */

/** The voice IDs. This version ships one voice. */
export type VoiceId = 'labzero-ai/zero-talk-11m';

/** The voice IDs this version ships, as an array. */
export const voices: readonly VoiceId[];

/** Options for loadVoice. */
export interface LoadOptions {
  /**
   * The engine. 'auto' (the default) uses WebGPU where the browser offers it on a secure page, and WebAssembly otherwise or when the
   * GPU fails. 'gpu' fails when WebGPU is not available. 'wasm' always uses WebAssembly. Node.js uses 'wasm' (its 'auto' too).
   */
  engine?: 'auto' | 'gpu' | 'wasm';
  /**
   * Where the three files (engine.wasm, model.bin, frontend.bin) are, instead of the package's own folder. In a browser, a URL of the
   * folder. In Node.js, a folder path, a file: URL or an http(s) URL of the folder. The folder's files.json is read from there too.
   */
  baseUrl?: string | URL;
}

/** One piece of audio from voice.stream(). */
export interface Chunk {
  /** Mono float samples in the range -1 to 1, at 24000 per second. Empty when `restart` is true. */
  samples: Float32Array;
  /**
   * True when the GPU engine failed part-way. The text then starts again on WebAssembly: drop the chunks you have; the chunks that come
   * after this one replace them. Only the engine's own failure does this; the text itself is never changed.
   */
  restart?: boolean;
}

/** The result of voice.speak(). */
export interface Audio {
  /** Mono float samples in the range -1 to 1. */
  readonly samples: Float32Array;
  /** 24000. */
  readonly sampleRate: 24000;
  /** samples.length / sampleRate. */
  readonly seconds: number;
  /**
   * Warnings from the engine, one per problem: for example, text it could not read (other scripts are left out). An empty array when
   * there are none.
   */
  readonly warnings: string[];
  /** The audio as a 16-bit mono WAV file (24000 Hz), as bytes. Synchronous. */
  toWav(): Uint8Array;
  /** The audio as an Ogg Opus file, as bytes. Smaller than WAV. Made by the engine. */
  toOgg(): Promise<Uint8Array>;
  /**
   * Writes the audio to a file and resolves when it is written: .wav is a 16-bit mono WAV, .ogg is Ogg Opus. Any other extension rejects
   * and says so. Node.js only: in a browser this rejects; use toWav() or toOgg() and offer the bytes as a download.
   */
  save(path: string): Promise<void>;
  /** Plays the audio in a browser and resolves when it has ended. In Node.js this rejects: write the file with save() instead. */
  play(): Promise<void>;
}

/** A loaded voice. */
export interface Voice {
  /** The engine in use: 'gpu' (WebGPU) or 'wasm' (WebAssembly). */
  readonly engine: 'gpu' | 'wasm';
  /** Makes the speech for the text. Resolves when the whole text is made. */
  speak(text: string): Promise<Audio>;
  /**
   * Makes the speech piece by piece, as an async iterable of chunks. Use it with `for await`. On WebAssembly the whole text is one chunk.
   * The chunks concatenated are the same audio that speak() returns.
   */
  stream(text: string): AsyncGenerator<Chunk, void, unknown>;
  /** Plays the text in a browser as it is made, and resolves when it has ended. In Node.js this rejects. */
  play(text: string): Promise<void>;
}

/**
 * Loads a voice: starts the engine and reads the model files. Rejects with an error that names the problem, for example an unknown id
 * (the error lists the valid ids) or an engine that is not available.
 * @param id The voice ID: 'labzero-ai/zero-talk-11m'.
 * @param options Engine and file location, both optional.
 */
export function loadVoice(id: VoiceId, options?: LoadOptions): Promise<Voice>;
