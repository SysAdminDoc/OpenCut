# Caption Line Breaking

OpenCut wraps subtitle text before it reaches SRT/VTT/ASS export, burn-in, or
styled overlay layout. The F242 gate replaces whitespace-only splitting with a
Unicode-aware caption breaker for CJK text that commonly arrives without spaces.

## Policy

- Prefer ordinary whitespace breakpoints for Latin-script captions.
- Allow breakpoints between CJK ideographs, kana, and hangul syllables.
- Keep closing punctuation, Japanese small kana, long-vowel marks, combining
  marks, and joiners attached to the neighboring glyph.
- Keep the Python backend dependency-light. ICU4X line segmentation is the
  reference model, but the source install does not take a mandatory binary ICU
  dependency.

## Implementation

- `opencut/core/caption_line_breaks.py` owns the shared line breaker.
- `opencut/export/srt.py` uses it for SRT and VTT text wrapping and for
  proportional cue splitting when word timestamps are absent.
- `opencut/core/styled_captions.py` uses CJK-aware layout tokens for overlay
  rendering.
- `opencut/core/subtitle_shot_aware.py` uses the same wrapper for timing-profile
  exports.
- `tests/test_caption_line_breaks.py` is wired into release smoke.

## References

- ICU4X line segmenter documentation: `https://docs.rs/icu/latest/icu/segmenter/index.html`
- Unicode Line Breaking Algorithm (UAX #14): `https://www.unicode.org/reports/tr14/`
