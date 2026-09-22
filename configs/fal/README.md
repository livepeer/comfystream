# Fal single-shot capabilities (ComfyStream)

Vendored from `runner-app-examples/api-proxy` with app IDs rewritten to
`comfystream/fal-<capability>`. `schema.json` bytes are unchanged so
`schema_sha256` matches the previous fleet.

- `index.json` — route identities
- `contracts/<capability>/route.json` + `schema.json`
- `overlay.yaml` / `overlay.json` — operator policy (capacity, deadline, sell price)

Refresh contracts:

```sh
python scripts/vendor_fal_catalog.py
```

## Orchestrator cutover

Stop advertising the old static `livepeer-example/fal-*` entries. On
`ai-realtime-go-livepeer-1`, the `-liveRunnerConfig` file is
`/opt/fal-adapter/runners.json` (mounted at `/config/runners.json`).

Keep only `livepeer-example/flux-klein` in that file (backup first). The
73 fal adapters are replaced by dynamic `register_runner` from the
ComfyStream live-runner (`comfystream/fal-*`). Dual registration of both
namespaces is out of scope — restart the orch after rewriting the file.

The ComfyStream live-runner dynamically `register_runner`s each
`comfystream/fal-*` app as `mode=single-shot` with `unit=fixed`.

Node install for this path uses `configs/nodes-live-runner.yaml`
(stream-pack + fal-api only), not the full `configs/nodes.yaml`.