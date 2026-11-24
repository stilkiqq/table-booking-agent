"""Voice messages (the optional part of the task), all local and free.

    speak    text -> Telegram voice note (OGG/Opus)
             Chatterbox (expressive, GPU, its own Python environment) with Kokoro-82M as the fallback
    listen   voice note -> text, faster-whisper

When the agent uses voice (set in the app, or VOICE_START / VOICE_THEN in .env):
    start with voice   yes / no: the first message is a voice note
    then               mirror: a voice note when the manager's last message was one | voice | text
    If the manager asks for text ("can you type it?") or for a voice note, the agent does that for the rest of the chat.
A voice note always carries the same text as its caption, so the chat (and its screenshot) stays readable.

Kokoro files: KOKORO_DIR (default ~/.cache/hyperframes/tts) with models/kokoro-v1.0.onnx and voices/voices-v1.0.bin,
from https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0).
Chatterbox: CHATTERBOX_PYTHON = python.exe of an environment with `pip install chatterbox-tts` (torch, a GPU helps).
Its timbre is copied from a reference clip, made once with Kokoro's British voice (voice_ref.wav).

Try it:  python voice.py "Good afternoon! A table for two at 17:00, please."    -> writes voice_test.ogg
         python voice.py --listen some.ogg
"""

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).parent
KOKORO_DIR = Path(os.getenv("KOKORO_DIR") or Path.home() / ".cache" / "hyperframes" / "tts")
VOICE = os.getenv("VOICE") or "bm_george"  # Kokoro British voices: bm_george, bm_lewis, bm_daniel, bf_emma, ...
WHISPER_MODEL = os.getenv("WHISPER_MODEL") or "small.en"
CHATTERBOX_PYTHON = os.getenv("CHATTERBOX_PYTHON") or next(
    (str(p) for p in [Path.home() / ".cache" / "yt3-chatterbox-rocm" / "Scripts" / "python.exe",
                      Path.home() / ".cache" / "gcg-chatterbox" / "Scripts" / "python.exe"] if p.exists()), "")
LIVELY = {"exaggeration": 0.8, "cfg_weight": 0.4}  # Chatterbox "lively": chosen by ear on 24.11
REF_FILE = HERE / "voice_ref.wav"
REF_TEXT = ("Good afternoon! I was wondering if you might have a table for two this evening at five o'clock? "
            "That would be lovely. Many thanks, and have a good day.")

THEN = {"mirror": "Like the manager", "voice": "Voice", "text": "Text"}
ENGINES = {"chatterbox": "Chatterbox", "kokoro": "Kokoro"}

_kokoro = None
_whisper = None
_worker: subprocess.Popen | None = None
_lock = threading.Lock()


# ---------- settings ----------

def _settings() -> dict:
    import llm  # settings.json is shared with the model choice
    return llm.settings()


def start_with_voice() -> bool:
    env = os.getenv("VOICE_START")
    return env.lower() in ("1", "true", "yes") if env else bool(_settings().get("voice_start", False))


def then() -> str:
    t = (os.getenv("VOICE_THEN") or _settings().get("voice_then") or "mirror").lower()
    return t if t in THEN else "mirror"


def engine() -> str:
    e = (os.getenv("TTS_ENGINE") or _settings().get("tts_engine") or "chatterbox").lower()
    return "chatterbox" if e == "chatterbox" and CHATTERBOX_PYTHON else "kokoro"


ASKS_TEXT = re.compile(r"\b(text|type|write)\b[^.?!]{0,20}\b(instead|rather)\b|\btype it\b|\b(in|by) (text|writing)\b|"
                       r"(can'?t|cannot|unable to) (listen|hear|play)|\bno voice (notes?|messages?)|rather read", re.I)
ASKS_VOICE = re.compile(r"(send|leave|record)\b.{0,15}\bvoice|voice (note|message)s? (please|is fine|instead)|"
                        r"rather (hear|listen)", re.I)


def manager_asks(text: str) -> str:
    """'text' / 'voice' if the manager asks for that way of talking, else ''."""
    t = text.replace("’", "'")
    return "text" if ASKS_TEXT.search(t) else "voice" if ASKS_VOICE.search(t) else ""


def ffmpeg() -> str:
    found = os.getenv("FFMPEG") or shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg not found: install it (winget install Gyan.FFmpeg) or set FFMPEG in .env")
    return found


# ---------- text -> speech ----------

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _clock(m: re.Match) -> str:
    h, mins = int(m.group(1)), int(m.group(2))
    part = "a.m." if h < 12 else "p.m."
    h12 = h % 12 or 12
    return f"{h12} {part}" if mins == 0 else f"{h12}:{mins:02d} {part}"


def speakable(text: str) -> str:
    """What a person would say out loud: "17:00" -> "5 p.m.", phone digit by digit, no emoji or brackets."""
    t = re.sub("[\U0001F000-\U0001FFFF☀-➿️]", "", text)
    t = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:am|pm|a\.m|p\.m))", _clock, t, flags=re.I)
    t = re.sub(r"\s*[()]\s*", ", ", t)
    t = re.sub(rf"\b(\w+day) (\d{{1,2}}) ({MONTHS})\b",
               lambda m: f"{m.group(1)} the {_ordinal(int(m.group(2)))} of {m.group(3)}", t)
    # A phone number is read digit by digit, "0" as "oh", like in Britain
    t = re.sub(r"\b0\d{3,4}[ \d]{5,8}\d\b", lambda m: ", ".join(
        " ".join("oh" if c == "0" else c for c in group) for group in m.group(0).split()), t)
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = t.replace("..", ".")
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r",\s*([,.!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip(" ,")


def _kokoro_wav(text: str, wav: Path) -> None:
    global _kokoro
    import numpy as np
    import soundfile as sf

    if _kokoro is None:
        import kokoro_onnx
        _kokoro = kokoro_onnx.Kokoro(str(KOKORO_DIR / "models" / "kokoro-v1.0.onnx"),
                                     str(KOKORO_DIR / "voices" / "voices-v1.0.bin"))
    samples, rate = _kokoro.create(text, voice=VOICE, speed=1.0, lang="en-gb")
    sf.write(wav, np.asarray(samples, dtype=np.float32), rate, subtype="PCM_16")


def _start_worker() -> subprocess.Popen:
    """Chatterbox in its own environment, loaded once and kept running."""
    global _worker
    if _worker and _worker.poll() is None:
        return _worker
    if not REF_FILE.exists():
        _kokoro_wav(REF_TEXT, REF_FILE)
    _worker = subprocess.Popen([CHATTERBOX_PYTHON, str(HERE / "chatterbox_worker.py"), str(REF_FILE)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, encoding="utf-8", bufsize=1)
    line = "?"
    while line and line.strip() != "ready":  # the library prints its own lines (e.g. "loaded PerthNet") first
        line = _worker.stdout.readline()
    if not line:
        _worker.kill()
        raise RuntimeError("Chatterbox did not start (see chatterbox_worker.py)")
    return _worker


def _answer(w: subprocess.Popen) -> dict:
    """The worker's next JSON line, skipping anything else the library prints."""
    while line := w.stdout.readline():
        if line.startswith("{"):
            return json.loads(line)
    return {"ok": False, "error": "worker stopped"}


def warm_up() -> None:
    """Load the voice models in advance, so the first voice note is not late. Errors only mean fallback later."""
    try:
        if engine() == "chatterbox":
            with _lock:
                _start_worker()
        listen_ready()
    except Exception as e:
        print(f"  (voice warm-up: {e})", flush=True)


def _chatterbox_wav(text: str, wav: Path) -> None:
    with _lock:
        w = _start_worker()
        w.stdin.write(json.dumps({"text": text, "out": str(wav), **LIVELY}) + "\n")
        w.stdin.flush()
        answer = _answer(w)
    if not answer.get("ok"):
        raise RuntimeError(answer.get("error"))


def _same(said: str, heard: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("p.m.", "pm").replace(" oh ", " 0 "))
    return difflib.SequenceMatcher(None, norm(said), norm(heard)).ratio()


def speak(text: str, out: str | Path | None = None) -> tuple[Path, float, str]:
    """Text -> OGG/Opus voice note. Returns (file, seconds, engine used).

    Chatterbox now and then mumbles a short phrase, so its result is checked with Whisper: one retry, then Kokoro.
    """
    import soundfile as sf

    say = speakable(text)
    if out is None:  # close the handle at once: on Windows an open file can't be overwritten or deleted
        fd, name = tempfile.mkstemp(suffix=".ogg", prefix="agent-voice-")
        os.close(fd)
        out = name
    out = Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        wav, used = Path(tmp) / "voice.wav", "kokoro"
        if engine() == "chatterbox":
            for attempt in range(2):
                try:
                    _chatterbox_wav(say, wav)
                    score = _same(say, listen(wav))
                    if score >= 0.8:
                        used = "chatterbox"
                        break
                    print(f"  (Chatterbox unclear, {score:.2f} match, {'retrying' if not attempt else 'using Kokoro'})",
                          flush=True)
                except Exception as e:
                    print(f"  (Chatterbox failed: {e}; using Kokoro)", flush=True)
                    break
        if used == "kokoro":
            _kokoro_wav(say, wav)
        seconds = sf.info(wav).duration
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libopus", "-b:a", "32k",
                        "-ac", "1", "-application", "voip", str(out)], check=True)
    return out, seconds, used


# ---------- speech -> text ----------

def listen_ready():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, compute_type="int8")
    return _whisper


def listen(path: str | Path, context: str = "") -> str:
    """A voice note (any format ffmpeg/PyAV reads, e.g. Telegram's .oga) -> English text.

    context: what the talk is about, e.g. "Booking a table for two at 17:00 today." Whisper uses it as a hint, which
    matters for numbers: without it "a table for two available at 17" once came out as "a table 4-2 variable at 16".
    """
    segments, _ = listen_ready().transcribe(str(path), language="en", vad_filter=True,
                                            initial_prompt=context or None)
    return " ".join(s.text.strip() for s in segments).strip()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--listen":
        print(listen(sys.argv[2]))
    else:
        phrase = " ".join(sys.argv[1:]) or "Good afternoon! I was wondering if you might have a table for two today at 17:00?"
        print("say:", speakable(phrase))
        f, sec, used = speak(phrase, HERE / "voice_test.ogg")
        print(f"{f} ({sec:.1f}s, {used}); heard back: {listen(f)}")
