"""Accept go-livepeer 0.9.x 402.orchestrator as LivePaymentChallenge.payment_url.

ja/v1.0.0 clients expect a full payment endpoint URL; orch returns the orch base
as ``orchestrator``, so map it to ``{orchestrator}/payment``.
"""

from __future__ import annotations

import glob
from pathlib import Path

matches = glob.glob(
    "/workspace/miniconda3/envs/comfystream/lib/python*/site-packages/livepeer_gateway/live_runner.py"
)
if not matches:
    raise SystemExit("livepeer_gateway.live_runner not installed")
path = Path(matches[0])
text = path.read_text(encoding="utf-8")
if "orchestrator.rstrip" in text and "/payment" in text:
    print(f"already patched {path}")
    raise SystemExit(0)

needle = '    payment_url = data.get("payment_url")\n'
if needle not in text:
    raise SystemExit(f"payment_url assignment not found in {path}")

insert = (
    needle
    + "    if not isinstance(payment_url, str) or not payment_url:\n"
    + '        orchestrator = data.get("orchestrator")\n'
    + "        if isinstance(orchestrator, str) and orchestrator.strip():\n"
    + '            payment_url = orchestrator.rstrip("/") + "/payment"\n'
)
text = text.replace(needle, insert, 1)
path.write_text(text, encoding="utf-8")
print(f"patched {path}")
