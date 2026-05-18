# Review Bundles

OpenCut review bundles are local-first zip archives built by
`POST /review/bundle` and `opencut.core.review_bundle.build_review_bundle`.
They now carry two marker surfaces:

- `markers.json` preserves the original OpenCut marker/comment payload for
  backward compatibility with existing automation.
- `markers.otio` is an OpenTimelineIO `Timeline.1` sidecar whose review notes
  are serialized as `Marker.2` objects on a timeline gap.

The OTIO sidecar uses `marked_range`, `color`, and an `opencut` metadata
namespace so downstream NLE or pipeline tools can import the timing anchor
without understanding the legacy OpenCut JSON shape. Comments remain in
metadata instead of the top-level marker `comment` property so the file stays
compatible with the older `opentimelineio>=0.15` floor supported by the
`opencut[otio]` extra.

Reference contracts:

- OpenTimelineIO Marker objects carry `marked_range`, `color`, `metadata`, and
  in newer releases `comment`: https://opentimelineio.readthedocs.io/en/latest/api/python/opentimelineio.schema.html
- The OTIO serialized data model lists `Marker.2` fields and versioned schema
  names: https://otio-website.vercel.app/docs/data-model/serialized-data-fields-only
- OTIO recommends application-owned metadata namespaces for data that is not
  first-class in the schema: https://github-wiki-see.page/m/AcademySoftwareFoundation/OpenTimelineIO/wiki/OpenTimelineIO-Application-Integrator%27s-Guide
