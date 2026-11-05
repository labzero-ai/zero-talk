"""Type stubs and documentation for zero_talk: Zero Talk 11M, an English text-to-speech model.

The package is the compiled engine with the model inside. Nothing is downloaded and nothing runs over the network.
"""

import os
from typing import Dict, List, Optional, TypedDict, Union, final, type_check_only

_PathLike = Union[str, "os.PathLike[str]"]

__all__ = ["Audio", "Chunk", "Stream", "Voice", "__version__", "load_voice", "model_info", "voices"]

__version__: str
"""The version of the package, for example "1.0.0"."""

@type_check_only
class _ModelFile(TypedDict):
    """One model file: its size in bytes, and its SHA-256 hash as lowercase hexadecimal."""

    bytes: int
    sha256: str

@type_check_only
class _ModelInfo(TypedDict):
    """What ``model_info`` returns: the voice identifier, and its files keyed by file name (``model.bin``, ``frontend.bin``)."""

    id: str
    files: Dict[str, _ModelFile]

def load_voice(id: str, model_dir: Optional[_PathLike] = None) -> Voice:
    """Load a voice and return it, ready for ``speak`` and ``stream``.

    ``id`` must be ``"labzero-ai/zero-talk-11m"``, the only voice in this version; any other value raises ``ValueError`` that lists the
    valid identifiers. Without ``model_dir``, the model files built into the package are used. With ``model_dir``, the model is read
    from that folder, which must hold ``model.bin`` and ``frontend.bin``; a missing file raises ``FileNotFoundError``.

    Loading takes a moment; the GIL is released while it runs.
    """

def voices() -> List[str]:
    """Return the identifiers of the voices that ``load_voice`` accepts: ``["labzero-ai/zero-talk-11m"]``."""

def model_info(id: str) -> _ModelInfo:
    """Return the sizes and SHA-256 hashes of the model files of voice ``id``, as the model page lists them.

    The result is a dict: ``{"id": ..., "files": {"model.bin": {"bytes": ..., "sha256": ...}, "frontend.bin": {...}}}``. An unknown
    ``id`` raises ``ValueError``.
    """

@final
class Audio:
    """Speech from ``Voice.speak``: mono samples at 24 kHz, and the warnings about what was left out of the text.

    Write it with ``save(path)``, or get the bytes with ``to_wav()`` or ``to_ogg()``.
    """

    @property
    def samples(self) -> memoryview:
        """The samples: a read-only ``memoryview`` of float32 (format ``"f"``), mono, from -1 to 1. ``numpy.asarray(audio.samples)``
        reads the same memory without a copy."""

    @property
    def sample_rate(self) -> int:
        """The sample rate in hertz: always 24000."""

    @property
    def seconds(self) -> float:
        """The length of the speech in seconds."""

    @property
    def warnings(self) -> List[str]:
        """What was left out of the text, one message per problem: words that could not be read, and characters the model has no
        sound for. An empty list when everything was spoken."""

    def save(self, path: _PathLike) -> None:
        """Write the speech to a file. The extension picks the format: ``.wav`` (16-bit PCM WAV) or ``.ogg`` (Ogg Opus, 24 kbit/s).

        Both carry an AI-generated marker in their metadata. Another extension raises ``ValueError``. A file that cannot be written
        raises ``OSError``.
        """

    def to_wav(self) -> bytes:
        """Return the speech as a complete WAV file in memory: 16-bit PCM, mono, 24 kHz. Its metadata carries an AI-generated marker."""

    def to_ogg(self) -> bytes:
        """Return the speech as a complete Ogg Opus file in memory: 24 kbit/s, mono, 24 kHz. Its metadata carries an AI-generated marker."""

@final
class Chunk:
    """A piece of the speech of a text, from ``Stream``: mono samples at 24 kHz."""

    @property
    def samples(self) -> memoryview:
        """The samples of this piece: a read-only ``memoryview`` of float32 (format ``"f"``), mono, from -1 to 1."""

@final
class Stream:
    """The speech of a text as it is made, one ``Chunk`` at a time. Get one from ``Voice.stream``.

    Iterate it: each chunk is made when it is reached, so the first audio is ready before the whole text is spoken. Concatenated, the
    chunks are the samples that ``Voice.speak`` gives for the same text.
    """

    def __iter__(self) -> Stream:
        """Return the stream itself."""

    def __next__(self) -> Chunk:
        """Make and return the next chunk. Raises ``StopIteration`` at the end of the text."""

@final
class Voice:
    """A voice: a model loaded and ready to speak. Get one from ``load_voice``.

    A voice can be used from several threads. Its calls release the GIL while the engine works.
    """

    def speak(self, text: str) -> Audio:
        """Speak a text and return the speech as an ``Audio``.

        The text is English: plain sentences, with numbers, dates and prices written as digits read out in words. Words that cannot
        be read, and characters the model has no sound for, are left out and named in ``Audio.warnings``. A text with nothing to say
        raises ``ValueError``.
        """

    def stream(self, text: str) -> Stream:
        """Start speaking a text piece by piece and return a ``Stream`` of ``Chunk`` objects.

        A text with nothing to say raises ``ValueError`` here, before the first chunk.
        """
