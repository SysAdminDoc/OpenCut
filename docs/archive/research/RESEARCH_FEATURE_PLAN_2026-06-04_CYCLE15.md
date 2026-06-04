# OpenCut Research Feature Plan - 2026-06-04 Cycle 15

Planning-only researcher artifact. This file captures Docker runtime parity
documentation and port-exposure drift. It does not modify source, tests,
workflows, generated files, or canonical planning docs.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-25, F001-F272, Waves L-T, and `docs/archive/research/` through Cycle 14.
- Primary evidence: `Dockerfile`, `docker-compose.yml`, README Docker launch
  docs, WebSocket bridge defaults, and current project context.

## Researcher Queue (Cycle 15 - 2026-06-04)

- [x] `docker-runtime-parity-refresh-2026-06-04` - checked Docker quick-start
  comments and compose port exposure against the current non-root image and
  documented runtime surface. `Dockerfile` now creates and runs as an `opencut`
  user with `HOME=/home/opencut`, and `docker-compose.yml` correctly mounts
  `opencut-data:/home/opencut/.opencut`, but the Dockerfile's own quick-start
  examples still mount `opencut-data:/root/.opencut`. The image declares
  `EXPOSE 5679 5680`, and README/PROJECT_CONTEXT document the WebSocket bridge
  on port 5680, but both compose services publish only `5679:5679`. The
  `/system/websocket/start` route calls `init_bridge(port=...)` without a host
  override, so the bridge's default `127.0.0.1` bind is also container-local if
  started inside Docker. Candidate RA-26 should decide whether Docker is HTTP
  only or panel/WebSocket capable, then align examples, ports, bind host, and
  tests accordingly.

## Quick Wins

- [ ] **P2 - Candidate RA-26 Align Docker runtime docs, volume home, and
  WebSocket exposure** - Why: Docker is advertised as a supported launch path,
  while OpenCut's product docs describe real-time WebSocket progress on 5680.
  Today the Dockerfile quick-start points users at the wrong persistence path
  for the non-root runtime user, and compose does not publish the WebSocket port
  that the image exposes and docs advertise. Evidence: `Dockerfile` run
  comments mount `/root/.opencut`; the final image creates `opencut` with
  `/home/opencut` and sets `HOME=/home/opencut`; `docker-compose.yml` mounts
  `/home/opencut/.opencut` but publishes only `5679:5679`; `Dockerfile` exposes
  `5679 5680`; README lists Docker support and WebSocket bridge port 5680;
  `opencut.core.ws_bridge.WebSocketBridge` defaults to host `127.0.0.1` and
  port 5680, and `/system/websocket/start` does not override that host. Touches:
  `Dockerfile`, `docker-compose.yml`, Docker/README docs, and a container
  config drift test. Acceptance: either Docker is explicitly documented as HTTP
  only, or compose/direct-run examples expose and bind the WebSocket bridge in a
  deliberate way; all quick-start volume examples use `/home/opencut/.opencut`;
  tests fail if Dockerfile examples, compose mounts, exposed ports, and
  documented runtime ports diverge again. Verify: focused Docker config test
  plus a Docker compose smoke that confirms `/health` and the chosen WebSocket
  posture. Complexity: S-M.

## Self-Audit

- Net-new check: RA-25 covers Docker dependency installs; this item covers
  Docker runtime docs, volume path, and port exposure.
- Net-new check: existing compose comments know `/root/.opencut` is wrong, but
  the Dockerfile quick-start comments still point users there.
- Risk calibration: this is a packaging/user-onboarding correctness issue, not
  a source-code data-loss bug in non-Docker installs.
- Lane-separation check: no implementation files or canonical docs were changed
  by this research pass.
