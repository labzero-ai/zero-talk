import type * as API from '../index.js' with { 'resolution-mode': 'import' };

export type { VoiceId, LoadOptions, Chunk, Audio, Voice } from '../index.js' with { 'resolution-mode': 'import' };
export declare const voices: typeof API.voices;
export declare const loadVoice: typeof API.loadVoice;
