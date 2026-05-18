# Local Review Portal

OpenCut's F231 review portal turns an existing review link into a short-lived
LAN share URL:

1. Create a review through `POST /review/create`.
2. Call `POST /review/portal/share` with `review_id`, optional `host`, `port`,
   `ttl_seconds`, `scheme`, and `service_name`.
3. Share the returned `/review/portal/<review_id>?expires=...&sig=...` URL.

The URL is signed with HMAC-SHA256 using the existing review token as the
server-side secret. The bearer URL expires at the requested timestamp and does
not expose the raw token.

The share response also includes:

- `caddyfile` — a Caddy reverse-proxy snippet for exposing the local Flask
  server on a LAN name or port.
- `mdns` — an `_http._tcp.local.` service descriptor for desktop launchers or
  installer-managed sidecars that publish Bonjour/mDNS records.

OpenCut does not start Caddy or publish mDNS records from the Flask request
handler. That keeps tests deterministic and avoids silently broadening network
exposure; packaging can wire the descriptors to a managed sidecar when a user
explicitly enables LAN review.
