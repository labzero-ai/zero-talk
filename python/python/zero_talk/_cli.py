"""The `zero-talk` command: the engine's own command line, run inside this package."""

import sys

from ._native import _run_cli


def main() -> None:
    """Entry point of the `zero-talk` console script."""
    sys.exit(_run_cli(sys.argv[1:]))
