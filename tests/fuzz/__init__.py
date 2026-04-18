"""Atheris-based fuzz tests for OpenCut parsers.

See ``test_parser_fuzz.py`` for the entry points.  These tests are
opt-in via the ``RUN_FUZZ=1`` env var — they don't run in the default
pytest pass because Atheris fuzz targets are infinite loops by
design.
"""
