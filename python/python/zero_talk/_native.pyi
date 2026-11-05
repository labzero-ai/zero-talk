"""The native implementation; public types are documented in zero_talk."""

from typing import List

from . import (
    Audio as Audio,
    Chunk as Chunk,
    Stream as Stream,
    Voice as Voice,
    __version__ as __version__,
    load_voice as load_voice,
    model_info as model_info,
    voices as voices,
)

__all__ = ["__version__", "load_voice", "voices", "model_info", "_run_cli", "Voice", "Audio", "Chunk", "Stream"]

def _run_cli(args: List[str]) -> int:
    """Run the command with arguments excluding the program name; return its exit status."""
