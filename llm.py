"""One function to call the language model, whichever one is chosen.

    ollama   a local model through Ollama, e.g. qwen3:14b (THINK=false in .env for qwen3)
    gemini   Google Gemini API, free tier: key from aistudio.google.com in .env as GEMINI_API_KEY

The choice is made in the app and saved to settings.json; environment variables (LLM, MODEL) override it.
"""

import asyncio
import json
import os
import time
from pathlib import Path

from pydantic import BaseModel

SETTINGS_FILE = Path(__file__).parent / "settings.json"
PROVIDERS = {"ollama": "Local (Ollama)", "gemini": "Gemini API"}
DEFAULT_MODEL = {"ollama": "qwen3:14b", "gemini": "gemini-3.5-flash"}


def settings() -> dict:
    if SETTINGS_FILE.exists():
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    return {}


def provider() -> str:
    return (os.getenv("LLM") or settings().get("llm") or "ollama").lower()


def model_name() -> str:
    p = provider()
    env = os.getenv("MODEL") if p == "ollama" else os.getenv("GEMINI_MODEL")
    return settings().get(f"{p}_model") or env or DEFAULT_MODEL[p]


def save(**values) -> None:
    SETTINGS_FILE.write_text(json.dumps({**settings(), **values}, indent=2), encoding="utf-8")


def local_models() -> list[str]:
    """Models installed in Ollama (empty if Ollama is not running)."""
    try:
        from ollama import Client
        return sorted(m.model for m in Client().list().models)
    except Exception:
        return []


async def chat_json(system: str, user: str, schema: type[BaseModel], temperature: float) -> str:
    """Ask the model for a JSON object matching `schema`. Returns the raw JSON text."""
    global last_model
    if provider() == "gemini":
        try:
            return await _gemini(system, user, schema, temperature)
        except Exception as e:  # every Gemini model busy / limit reached: don't leave the manager waiting
            local = settings().get("ollama_model") or os.getenv("MODEL") or DEFAULT_MODEL["ollama"]
            print(f"  (Gemini failed: {str(e)[:80]}; falling back to local {local})", flush=True)
            last_model = local
            return await _ollama(system, user, schema, temperature, local)
    last_model = model_name()
    return await _ollama(system, user, schema, temperature)


async def _ollama(system: str, user: str, schema: type[BaseModel], temperature: float, model: str = "") -> str:
    from ollama import AsyncClient  # local server, http://127.0.0.1:11434

    think = (os.getenv("THINK") or "").lower()
    extra = {} if not think else {"think": think == "true"} if think in ("true", "false") else {"think": think}
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    response = await AsyncClient().chat(model=model or model_name(), messages=messages, format=schema.model_json_schema(),
                                        options={"temperature": temperature}, **extra)
    return response.message.content


# Free-tier Gemini answers 503 "high demand" per model, often for minutes, while other models answer fine
# (checked 24.11: 3.8/3.7-flash busy every time, 3.6/3.5/3-flash-preview fine). So instead of waiting on a busy
# model we move down this list at once; the chosen model goes first.
GEMINI_FALLBACKS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite"]
BUSY_FOR = 120  # seconds a model that said 503/429 is skipped
_busy_until: dict[str, float] = {}
last_model = ""  # which model actually answered the last call (shown in the log and the app)


def gemini_chain() -> list[str]:
    return list(dict.fromkeys([model_name(), *GEMINI_FALLBACKS]))


async def _gemini(system: str, user: str, schema: type[BaseModel], temperature: float) -> str:
    global last_model
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],  # never hang on a busy server, switch model instead
                          http_options=types.HttpOptions(timeout=30_000, retry_options=types.HttpRetryOptions(attempts=1)))
    config = types.GenerateContentConfig(system_instruction=system or None, temperature=temperature,
                                         response_mime_type="application/json", response_schema=schema)
    last_error: Exception | None = None
    for round_ in range(2):  # second round: everything was busy, wait a little and try all again
        if round_:
            await asyncio.sleep(10)
        for model in gemini_chain():
            if round_ == 0 and _busy_until.get(model, 0) > time.time():
                continue
            try:
                response = await client.aio.models.generate_content(model=model, contents=user, config=config)
                if not response.text:
                    raise ValueError("empty answer")
                _busy_until.pop(model, None)
                if model != last_model and last_model:
                    print(f"  (Gemini: now answering {model})", flush=True)
                last_model = model
                return response.text
            except (errors.APIError, TimeoutError, OSError, ValueError) as e:
                code = getattr(e, "code", None) or type(e).__name__
                # 404 = model closed for this key, 400 = model can't do JSON schema: don't come back to it soon
                _busy_until[model] = time.time() + (3600 if code in (400, 404) else BUSY_FOR)
                print(f"  (Gemini {model}: {code}, trying next model)", flush=True)
                last_error = e
    raise RuntimeError(f"all Gemini models busy: {last_error}")
