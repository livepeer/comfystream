# Integrating sd-turbo and flux-klein

Both models are persistent Live Runner apps. A session is a video in and a video out. The request is a preset and a prompt. Do not send a Comfy workflow.

| Short name | App id | Presets | Extra fields |
| --- | --- | --- | --- |
| `sd-turbo` | `comfystream/sd-turbo` | `neon-stage`, `watercolor`, `anime` | `prompt`, `negative_prompt` |
| `flux-klein` | `livepeer-example/flux-klein` | `neon-stage`, `watercolor`, `cosmic` | `prompt`, `seed`, `input_blend` |

`seed` of `-1` picks new noise every frame. `input_blend` is the camera weight from 0 to 1. A field the app does not accept returns `unsupported_param`.

## Token

Set `TOKEN` to the base64 gateway token from PymtHouse (signer URL, `Authorization` header, discovery URL). Pass it as `--token` or leave `--token` unset and the client reads `TOKEN`. Do not commit the token.

The client uses `livepeer_gateway.token.parse_token` and then `reserve_session`. Token fields win over any signer or discovery URL you pass yourself. Use the `livepeer-python-gateway` checkout on `main` (1.0 or later).

## Run a clip

From a machine that can import `livepeer_gateway` and `av`:

```bash
export PYTHONPATH=/path/to/livepeer-python-gateway/src
export TOKEN='<your base64 gateway token>'

python server/realtime_client.py clip.mp4 \
  --app sd-turbo --preset neon-stage \
  --fps 15 --max-frames 60 --output sd-turbo-out.ts

python server/realtime_client.py clip.mp4 \
  --app flux-klein --preset cosmic \
  --fps 8 --max-frames 40 --output flux-klein-out.ts
```

`--app` accepts the short name or the full app id. `--prompt` replaces the preset text. `--reprompt 5=an oil painting` changes the prompt five seconds into the clip.

Cursor launch configs `Realtime: sd-turbo` and `Realtime: flux-klein` run the same script against `/tmp/e2e/clip.mp4`. Set `TOKEN` in your shell or VS Code environment before launching.

Play the result with `ffplay sd-turbo-out.ts`.

## What the client does

1. `parse_token(--token)` to get the signer and discovery URL, then resolve the orchestrator list. The signer drops the list when an app filter is added to that URL, so the client filters the app after discovery.
2. `reserve_session(app=...)`. The gateway pays through the signer for as long as the session is open.
3. `POST {app_url}/stream` with a flat JSON body, for example:

```json
{"preset": "neon-stage", "prompt": "a neon concert stage, lasers, haze"}
```

4. Publish the file to the returned `in` URL and write the `out` URL to disk.
5. `stop_runner_session` when the file ends, which releases the GPU session.

A warm runner returns `compute: "warm"` and `startup_s` near 0. A cold runner loads on that first `/stream` and can take about 15–35 seconds. `estimated_startup_s` on the response is the expected wait. `/stream` returns 503 `startup_exceeds_limit` if you also send `max_startup_s` and the estimate is higher.

`/stream` returns `in` and `out` trickle URLs plus `status`, `preset`, and `params`.

## Change, pause, and stop

These are `POST`s to the same `app_url`, while the session is open. The orchestrator adds the session header. You do not send it yourself.

| Call | Body | Effect |
| --- | --- | --- |
| `/update` | `{"prompt": "..."}` or `{"preset": "watercolor"}` | Changes the look. The video URLs stay the same. |
| `/pause` | `{"reason": "between songs"}` | Shows the fallback visual and skips inference. The session stays reserved. |
| `/resume` | `{}` | Back to live generation. |
| `/stop` | `{"reason": "booking ended"}` | Ends the session. A second stop returns the same record with `already_stopped: true`. |
| `GET /session` | | Live state, or the usage record after stop. |

`/stop` reports `room_id`, durations, frames, and a cost estimate when you passed metadata at start:

```json
{"preset": "cosmic", "metadata": {"room_id": "r12", "venue_id": "v3", "booking_id": "b9", "environment": "pilot"}}
```

`environment` is `test`, `pilot`, or `production`.

## Errors

Failures are `{"error": {"code", "message", "field"}}`.

| Code | When |
| --- | --- |
| `unsupported_param` | The field is not valid for that app, including a `workflow` object. |
| `unknown_preset` | Preset name is not one of the names above. |
| `invalid_param` | Empty prompt, or a number outside its range. |
| `invalid_metadata` | Bad room id or environment. |
| `startup_exceeds_limit` | Cold start is slower than `max_startup_s`. |
| `session_conflict` | That runner already has another session. |
| `runner_unavailable` | The pipeline failed to load. |
