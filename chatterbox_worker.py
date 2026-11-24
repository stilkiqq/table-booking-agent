"""Chatterbox TTS worker. Runs in Chatterbox's own Python environment (torch + chatterbox-tts), not the agent's.

voice.py starts it once and keeps it running, so the model loads only once:
    <chatterbox python> chatterbox_worker.py <reference.wav>
stdin:  one JSON per line  {"text": "...", "out": "file.wav", "exaggeration": 0.8, "cfg_weight": 0.4}
stdout: "ready" after loading, then one JSON per request  {"ok": true, "seconds": 4.2} or {"ok": false, "error": "..."}
"""

import json
import sys

import numpy as np
import soundfile as sf
import torch

torch.backends.cudnn.enabled = False  # AMD/ROCm on Windows: MIOpen batchnorm fails, torch's own kernels work
from chatterbox.tts import ChatterboxTTS  # noqa: E402


def device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    # With an AMD CPU+GPU torch sees both; the real card has more compute units
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).multi_processor_count)
    torch.cuda.set_device(best)
    return f"cuda:{best}"


def trim(s: np.ndarray, rate: int) -> np.ndarray:
    loud = np.where(np.abs(s) > 0.01)[0]
    pad = int(0.05 * rate)
    return s[max(0, loud[0] - pad): loud[-1] + pad] if len(loud) else s


def main() -> None:
    model = ChatterboxTTS.from_pretrained(device=device())
    model.prepare_conditionals(sys.argv[1])
    print("ready", flush=True)
    for line in sys.stdin:
        try:
            job = json.loads(line)
            wav = model.generate(job["text"], exaggeration=job.get("exaggeration", 0.8),
                                 cfg_weight=job.get("cfg_weight", 0.4))
            s = trim(wav.squeeze(0).cpu().numpy().astype(np.float32), model.sr)
            sf.write(job["out"], s, model.sr, subtype="PCM_16")
            print(json.dumps({"ok": True, "seconds": round(len(s) / model.sr, 2)}), flush=True)
        except Exception as e:  # report and keep serving
            print(json.dumps({"ok": False, "error": str(e)[:300]}), flush=True)


if __name__ == "__main__":
    main()
