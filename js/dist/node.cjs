// CommonJS bridge to the Node.js ES module. Loading a voice already returns a promise.
exports.voices = Object.freeze(['labzero-ai/zero-talk-11m']);
exports.loadVoice = async function loadVoice(id, options) {
  const engine = await import('./node.js');
  return engine.loadVoice(id, options);
};
