# OpenCut Research Report

Root synthesis of current research and planning inputs. Detailed research plans
are archived under [docs/archive/research](docs/archive/research/).

Last consolidated: 2026-06-04.

## Executive Summary

OpenCut is already extremely broad. The highest-value research direction is not
another large wave of model surfaces. It is making existing work easier to run,
debug, resume, extend, and trust.

The May 26 research pass identified the strongest v1.33+ opportunities. N1 is
now closed in `ROADMAP.md` v4.87; the remaining queue is tracked in
`ROADMAP.md` under "Active Continuation Queue (May 26 Plan)".

1. Content-addressable transcript cache by audio hash. **Shipped in v4.87.**
2. `missing_dependency()` responses that name the exact pip extra.
3. GPU semaphore acquire-wait behavior instead of instant contention failures.
4. Disk preflight on heavyweight render/model routes.
5. Resumable interrupted jobs.
6. `GET /webhooks/event-types` discovery.
7. Plugin background-job registration.
8. Third-party agent skill loading from the user data directory.
9. Rich job metadata such as peak VRAM, exit reason, and started-at fields.
10. Request-ID propagation into FFmpeg/subprocess stderr.

## Research Inputs

- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md) captured the governance, route-surface, agent, UXP, i18n, a11y, CI, and supply-chain loop that fed the current Unreleased work.
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-26.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-26.md) is the performance, observability, crash-recovery, plugin extensibility, resource-preflight, and trust-signals pass.
- [docs/RESEARCH.md](docs/RESEARCH.md) keeps the earlier tracked research summary.
- [ROADMAP.md](ROADMAP.md) remains the canonical detailed F-number and wave-letter ledger.
- [ROADMAP-NEXT.md](ROADMAP-NEXT.md) remains the wave-letter worksheet for older active-wave references.

## Planning Implications

| Theme | Implication |
|---|---|
| Performance | Cache expensive transcript/model outputs by stable content hashes before adding more user-facing surfaces. |
| Observability | Job metadata and request correlation must span Python, FFmpeg, subprocesses, routes, and panel state. |
| Recovery | Interrupted jobs should become resumable or explicitly non-resumable with useful reasons. |
| Extensibility | Plugins need safe background-job primitives and capability-scoped skill loading, not just Flask route registration. |
| Release trust | Keep generated manifests, smoke gates, docs-size checks, advisories, signing, SBOM, and package wiring in one release-readiness loop. |
| UXP migration | CEP remains supported, but UXP parity and WebView cutover work must stay visible because CEP end-of-life risk is time-bound. |

## Archive Notes

Root `research.md` is ignored by policy and was not tracked at the start of this
pass. It remains untouched as a local artifact. The tracked May 25 and May 26
research plans were moved into `docs/archive/research/`.
