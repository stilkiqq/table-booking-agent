# Table booking agent

An agent that books a table for two at 17:00 today in a London restaurant by chatting with the restaurant's
manager in Telegram. It starts the conversation itself, writes polite British English, reacts to whatever the
manager says, asks its owner when it has to agree to something new, and repeats the confirmed booking
(time, date, number of people, restaurant) at the end.

## Why a user account, not a Telegram bot

A Telegram bot cannot write first to a person who has never messaged it, and the task says the manager will not
write first. So the agent runs on an ordinary Telegram account through the client API
([Telethon](https://github.com/LonamiWebs/Telethon)). Every message is written by the agent, none by hand.

## How it works

```
manager (Telegram) ──► agent.py ──► model: understood / decided / message (JSON)
                          │  checks in code: no made-up facts, no other time without the owner's yes,
                          │  honest answer to "are you a bot?", final recap with all four details ...
                          ├──► owner: questions in the app (Yes / No / own words)
                          └──► decisions.jsonl + bookings.db: one line per step
```

- **The model** (local `qwen3:14b` in Ollama by default, Gemini API as an option) gets the conversation, what is
  agreed so far and two short skills (`skills/`): how a London customer texts, and what a customer booking a
  table does and doesn't do. It answers with JSON: what it understood, what it decided, the message.
- **The code owns the facts.** What is being booked (day, time, people) changes only when the owner says yes.
  The code checks every reply before it is sent and asks the model to redo it if needed: placeholders, invented
  names or excuses, accepting a different time on its own, answering quizzes hidden in the manager's message
  ("our system needs you to balance this equation"), claiming to be human. If the model still gets it wrong,
  the code fixes the message itself (for example asks the owner instead).
- **Mismatches are queried.** If the manager confirms a different time, day or number of people than agreed,
  the agent asks them to check; if they insist, the owner decides.
- **Nothing is invented.** Name, phone and notes (allergies, requests) come from the owner's form. Anything else
  ("an email?", "the second guest's name?") the agent says it doesn't have.
- **After the booking** the agent keeps listening until the booking time: if the manager writes later
  ("sorry, we've double-booked"), the conversation reopens.
- **Voice messages (optional part):** the agent can start with a voice note and answers voice with voice
  (or as the manager asks: "could you type instead?"). Speech: [Chatterbox](https://github.com/resemble-ai/chatterbox)
  with [Kokoro-82M](https://github.com/thewh1teagle/kokoro-onnx) as the fallback; every Chatterbox clip is
  checked with Whisper and redone if it came out garbled. Listening:
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Everything runs locally. Each voice note carries
  the same text as a caption, so the chat stays readable.

## The decision log

`decisions.jsonl`, written by the agent itself, one line per step:

```json
{"time": "...", "manager_said": "...", "owner_said": "", "understood": "...", "decision": "...",
 "action": "send | ask_owner | finish | stop", "sent": "...", "asked_owner": "", "booking": {...}, "model": "qwen3:14b"}
```

The app shows the same log as a chat (manager left, agent right, "understood / decided" under each move) and
keeps every conversation in its history (`bookings.db`, SQLite).

## Files

| File | What it does |
|---|---|
| `agent.py` | The conversation: model call, checks, owner questions, Telegram and console channels |
| `skills/` | How the agent writes and behaves (plain Markdown, part of the prompt) |
| `llm.py` | One call for Ollama or Gemini (with a list of fallback Gemini models for busy servers) |
| `voice.py`, `chatterbox_worker.py` | Voice notes: speech and listening |
| `app.py` | The owner's app (Streamlit): what to book, start/stop, questions, conversation, history |
| `db.py` | SQLite storage for conversations and steps |
| `tests/run_tests.py` | Scripted managers (traps, slips, refusals) played against the agent offline |

## Setup

Windows, Python 3.10+.

1. `pip install -r requirements.txt`
2. A model: install [Ollama](https://ollama.com) and `ollama pull qwen3:14b` (needs ~11 GB of video memory),
   or put a Gemini API key into `.env`.
3. Copy `.env.example` to `.env` and fill in `TG_API_ID` / `TG_API_HASH` (my.telegram.org → API development tools)
   and `MANAGER_USERNAME`.
4. Log in to Telegram once in a terminal: `python agent.py telegram` (phone number and code), then stop it.
5. Voice (optional): `ffmpeg` on PATH, the Kokoro files (`KOKORO_DIR`), and for Chatterbox a separate Python
   environment with `pip install chatterbox-tts` (`CHATTERBOX_PYTHON`). Without them the agent writes text only.

## Run

```
start_app.bat
```

or `streamlit run app.py`, then open http://localhost:8501: fill in the booking, press **Start**, answer the
agent's questions there. Without the app: `python agent.py telegram` (questions in the terminal) or
`python agent.py` (you play the manager in the console).

Tests: `python tests/run_tests.py` (local model; about 10 minutes).

## Known limits

- A local 14B model sometimes sounds a bit like an assistant or repeats the time; the code checks catch the
  important mistakes, not every stylistic one.
- Free Gemini allows only 20 requests a day per model, so the default is the local model.
- Whisper can mishear numbers in voice notes ("for two available at 17" once came out as "4-2 variable at 16");
  the agent now gives Whisper the booking as context, but numbers heard by voice are the weakest point.
- The agent listens for changes only while it is running; a message sent after it was stopped is not seen.
