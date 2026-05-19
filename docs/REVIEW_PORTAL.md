# Local Review Portal

OpenCut's F231/F232 review portal turns an existing review link into a
short-lived LAN or operator-managed cross-site share URL:

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
- `headscale` — optional. When the request includes a `headscale` object,
  OpenCut returns a Headscale/Tailscale operator plan for cross-site review.

OpenCut does not start Caddy or publish mDNS records from the Flask request
handler. That keeps tests deterministic and avoids silently broadening network
exposure; packaging can wire the descriptors to a managed sidecar when a user
explicitly enables LAN review.

## Optional Headscale Path

For cross-site review, `POST /review/portal/share` can include:

```json
{
  "review_id": "r1",
  "host": "opencut-review.local",
  "port": 8080,
  "headscale": {
    "url": "https://headscale.example.net",
    "user": "post-team",
    "machine_name": "opencut-client-review",
    "tags": ["opencut-review"],
    "ttl_hours": 24
  }
}
```

The response includes `headscale.commands.create_preauth_key` and
`headscale.commands.join_tailnet` arrays. They are command plans only; OpenCut
does not create Headscale keys, run `tailscale up`, store preauth keys, or
enable cross-site networking from a request handler. Keep the signed portal URL
TTL short because the URL remains the bearer credential.
