"""
Entry point for `python -m opencut`.

Routes to the CLI by default. The server has its own module entry:
  python -m opencut           -> CLI (silence, captions, full, info)
  python -m opencut.server    -> Backend server for the Premiere panel
"""

from opencut.cli import main

main()
