"""Zero Talk 11M: English text to speech, with the model inside the package.

Start with ``load_voice("labzero-ai/zero-talk-11m")``. The types and their documentation are in ``__init__.pyi``.
"""

from ._native import Audio, Chunk, Stream, Voice, __version__, load_voice, model_info, voices

__all__ = [
    "Audio",
    "Chunk",
    "Stream",
    "Voice",
    "__version__",
    "load_voice",
    "model_info",
    "voices",
]
