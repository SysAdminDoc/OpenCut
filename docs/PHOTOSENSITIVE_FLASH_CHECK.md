# Photosensitive Flash Check

OpenCut's `/accessibility/flash-detect` route is a standards-oriented preflight for photosensitive seizure risk. It is not a certified Harding FPA, PEAT, or broadcaster acceptance tool; it is designed to catch risky material early in local export QC and review workflows.

## Standards Basis

- ITU-R BT.1702-3 (11/2023) defines potentially harmful flashing images for television, including luminance flash pairs, the 25% screen-area condition, and the 360 ms / 334 ms safe-gap clarification for 50 Hz / 60 Hz environments: https://www.itu.int/rec/R-REC-BT.1702-3-202311-I
- W3C WCAG 2.2 Success Criterion 2.3.1 defines the web threshold as no more than three general or red flashes in any one-second period unless the flashing area is below the threshold: https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold
- Trace RERC PEAT documents the working saturated-red definition used by accessibility tooling: `R/(R+G+B) >= 0.8` and `(R-G-B)*320` delta greater than 20: https://trace.umd.edu/photosensitive-epilepsy-analysis-tool-peat-user-guide/
- TV Tokyo's English animation image-effect guidelines document the Japan-style animation rule of no more than one flash or sudden cut per one-third second and warn against isolated-red flashes: https://www.tv-tokyo.co.jp/kouhou/guideenglish.htm
- Ofcom Broadcasting Code Rule 2.12 requires television broadcasters to take precautions to keep photosensitive-epilepsy risk low: https://www.ofcom.org.uk/tv-radio-and-on-demand/broadcast-standards/section-two-harm-offence/

## Route

`POST /accessibility/flash-detect`

Request fields:

- `filepath`: source video path, handled by the async job wrapper.
- `max_flashes_per_sec`: defaults to `3`.
- `min_luminance_change`: defaults to `0.1`, matching the WCAG/BT.1702 relative-luminance threshold.
- `standard_profile`: `bt1702-3`, `wcag22`, or `japan-animation`.
- `screen_area_ratio`: estimated concurrent flashing area, default `1.0`. Values at or below `0.25` stay below the BT.1702/WCAG area failure condition.

Response additions:

- `standard_profile` and `frame_rate_profile`.
- `general_flash_count` and `red_flash_count`.
- `thresholds`, including the applied 50 Hz or 60 Hz safe-gap value.
- `events[*].flash_type`, `min_gap_ms`, `unsafe_gap_count`, `area_ratio`, and `standard`.

## Implementation Notes

The checker counts flashes as pairs of opposing transitions, not single frame-to-frame jumps. It uses FFmpeg `signalstats` for frame luminance and a 1x1 RGB average as a low-cost saturated-red preflight. Because this is a global-frame approximation, any failed result should be treated as actionable, while any pass for formal broadcast delivery should still be confirmed with a certified analyzer or the destination platform's required QC tool.
