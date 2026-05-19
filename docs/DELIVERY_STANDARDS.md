# Delivery Standards Planning

OpenCut exposes local-first planning presets for standards-heavy mastering
workflows. The routes return deterministic command arrays, validation notes, and
commercial boundaries. They do not run external tools, create platform
credentials, certify deliverables, or replace receiver-side QC.

## Routes

`GET /delivery/mastering-presets`

Lists the available delivery-standard presets and tool counts.

`GET /delivery/mastering-plan?preset=<id>`

Builds a read-only operator plan. Optional query parameters:

- `source_path`: source media path. When supplied through Flask, OpenCut validates
  it with the normal local path guard but does not require the file to exist.
- `output_dir`: output directory path. Also validated without creating files.
- `title`: human-readable package title.
- `dolby_profile`: `5` or `8.1` for the Dolby Vision OSS plan.
- `adm_render_layout`: compact EAR layout token such as `0+5+0`.

`POST /delivery/mastering-plan`

Accepts the same fields as JSON and is protected by the normal OpenCut CSRF
token. It remains a planning endpoint and has no file-system side effects.

## Presets

| Preset | F# | Purpose | Boundary |
|---|---:|---|---|
| `netflix_imf_dolby_vision` | F245 | Plans an IMF Application 2E handoff with Dolby Vision metadata checks, optional Bento4 proxy packaging, and Photon/Backlot validation notes. | Netflix acceptance, Dolby Vision licensing, and platform QC stay external. |
| `dpp_imf_broadcast` | F246 | Plans a DPP/BBC/ARD/EBU-oriented IMF handoff with DPP metadata sidecars and receiver-check notes. | Receiver-specific DPP, BBC, ARD, EBU, or facility rules remain authoritative. |
| `dolby_vision_profile_5_8_1` | F247 | Plans a dovi_tool plus Shaka/Bento4 review path for Dolby Vision Profile 5 or 8.1 assets. | Profile 7/FEL conversion is explicitly lossy/limited; final certified delivery needs licensed Dolby tooling. |
| `adm_bwf_atmos_master` | F248 | Plans ADM BW64 master preparation, axml/chna inspection, and EAR renderer QC. | Final Dolby Atmos `.ec3`/DD+JOC output remains a commercial Dolby Encoding Engine handoff. |

## Source Notes

- Netflix documents Dolby Vision IMF delivery expectations and Backlot IMF
  inspection stages in its Partner Help Center:
  <https://partnerhelp.netflixstudios.com/hc/en-us/articles/360000599948-Dolby-Vision-HDR-Mastering-Guidelines>
  and
  <https://partnerhelp.netflixstudios.com/hc/en-us/articles/115000614752-Backlot-Delivery-Instructions-for-IMF>.
- `dovi_tool` is the open Dolby Vision metadata utility used for RPU inspection,
  generation, export, editing, and HEVC metadata handling:
  <https://github.com/quietvoid/dovi_tool>.
- Shaka Packager and Bento4 are OSS packaging toolchains for DASH/HLS/CMAF-style
  review packages:
  <https://github.com/shaka-project/shaka-packager> and
  <https://www.bento4.com/documentation/>.
- DPP publishes public guidance for AS-11 and IMF delivery-requirements work:
  <https://www.thedpp.com/specs/imf>.
- EBU ADM guidance explains BW64 `axml` and `chna` chunks, and the EBU ADM
  Renderer provides open command-line QC utilities:
  <https://adm.ebu.io/reference/excursions/bw64_and_adm.html>,
  <https://adm.ebu.io/background/rendering.html>, and
  <https://github.com/ebu/ebu_adm_renderer>.

## Example

```bash
curl "http://127.0.0.1:5679/delivery/mastering-plan?preset=dolby_vision_profile_5_8_1&dolby_profile=8.1"
```

The response contains `runs_external_tools=false`, command `argv` arrays, source
links, validation notes, and the commercial boundary for the selected preset.
