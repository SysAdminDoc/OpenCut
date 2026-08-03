# Delivery Transfer Bundles

OpenCut can prepare a local delivery bundle for one-shot transfer tools without taking over the user's network or cloud configuration. The F234 transfer flow creates a zip archive, writes a machine-readable manifest, and returns ready-to-run command plans for croc and rclone.

## Routes

- `GET /delivery/transfer/options` returns the supported delivery transfer methods and whether each binary is available on `PATH`.
- `POST /delivery/transfer-bundle` creates an async job that packages one or more files or folders into a zip bundle and returns transfer commands.

The bundle job accepts:

- `paths` or `files`: one or more source files/folders to include.
- `filepath`: legacy single-source alias accepted by async job plumbing.
- `output_path`: exact zip destination.
- `output_dir` and `bundle_name`: used when `output_path` is omitted.
- `method` or `methods`: `croc`, `rclone`, `both`, or a list of methods.
- `croc_code`: optional fixed croc code.
- `croc_relay`: optional croc relay.
- `rclone_remote`: required when rclone is selected.
- `rclone_path`: optional remote subdirectory.

## Output

The job returns:

- `bundle_path`: created zip bundle.
- `manifest_path`: sibling `.transfer.json` manifest.
- `source_paths`, `source_count`, `total_source_bytes`, and `bundle_bytes`.
- `commands`: one entry per selected method, including `argv`, `shell_command`, availability, and notes.
- `warnings`: missing-tool notes when `croc` or `rclone` is not installed.

The zip also contains `delivery_transfer_manifest.json` with the same source-file inventory. External transfer tools are not run from request handling; users keep control over croc codes, relay choice, rclone remotes, credentials, and final send/copy execution.
