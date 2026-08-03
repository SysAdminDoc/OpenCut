# Review Notifications

OpenCut's F233 review notification surface adds two opt-in ways to watch review
activity without requiring a cloud service.

## Atom Feed

`GET /review/feed.atom` returns an Atom feed generated from saved review links.
Use `project_id`, `review_id`, and `limit` query parameters to narrow the feed:

```text
GET /review/feed.atom?project_id=client-a&limit=50
```

Entries are emitted for review status snapshots and timestamped review comments.
Reviews with no `project_id` are grouped under `default`.

## Signed Webhooks

`POST /api/webhooks` requires a non-empty `secret` field by default. For trusted
local testing only, callers may explicitly opt into unsigned delivery with
`POST /api/webhooks?allow_unsigned=true`; OpenCut writes
`~/.opencut/webhooks_unsigned.txt` once when unsigned webhook configs are
present.

Signed outbound deliveries include:

- `X-OpenCut-Signature: sha256=<hex>`
- `X-OpenCut-Signature-Algorithm: HMAC-SHA256`

The signature covers the exact JSON request body bytes. Existing unsigned
configs still load for compatibility. Listing webhooks returns `has_secret` and
never echoes the secret value.

Review comments and status updates emit best-effort events through the existing
webhook system:

- `review.comment_added`
- `review.status_changed`

Webhook failures are logged but do not block the review action.
