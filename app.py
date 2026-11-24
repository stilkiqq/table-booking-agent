"""The owner's app: what to book, starting the agent in Telegram, answering its questions, the conversation log.

Run: streamlit run app.py
"""

import html
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from datetime import time as dtime

import streamlit as st

import db
import llm
import voice
from agent import HERE, LOG_FILE, OWNER_FILE, REQUEST_FILE, BookingRequest, load_request

st.set_page_config(page_title="Table booking", page_icon="🍽️", layout="centered")
OUT_FILE = HERE / "agent_output.txt"
MANAGER = (llm.settings().get("manager") or os.getenv("MANAGER_USERNAME", "")).lstrip("@")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

st.title("🍽️ Table booking")

# ---------- 1. Booking ----------
req = load_request()
with st.container(border=True):
    st.subheader("1. Booking")
    # No form and no Save button: every change is saved at once, so the agent never starts with stale details
    c1, c2, c3 = st.columns(3)
    today = date.today()
    # A list of days, not st.date_input: its calendar greys out next month's days
    days = [today + timedelta(days=i) for i in range(60)]

    def day_label(d: date) -> str:
        prefix = {0: "Today, ", 1: "Tomorrow, "}.get((d - today).days, "")
        return f"{prefix}{d:%a %d %b %Y}"

    # Fixed keys: the fields keep what you typed (also an emptied field), filled from request.json once
    for k, v in {"f_day": req.day() if req.day() in days else today, "f_party": req.party_size, "f_name": req.name,
                 "f_phone": req.phone, "f_policy": req.if_unavailable, "f_notes": req.notes}.items():
        st.session_state.setdefault(k, v)
    day = c1.selectbox("Date", days, format_func=day_label, key="f_day")
    h, m = map(int, req.time.split(":"))
    st.session_state.setdefault("f_time", dtime(h, m))
    time_value = c2.time_input("Time", step=900, key="f_time")
    party = c3.number_input("Guests", min_value=1, max_value=20, key="f_party")
    c4, c5 = st.columns(2)
    name = c4.text_input("Name", key="f_name")
    phone = c5.text_input("Phone", key="f_phone")
    policy = st.radio("If that time is taken", ["ask_owner", "stop"],
                      format_func={"ask_owner": "Find the nearest option and ask me",
                                   "stop": "Say goodbye politely"}.get, key="f_policy")
    notes = st.text_area("Notes", height=80, placeholder="Allergies, seating, occasion…", key="f_notes")
    edited = BookingRequest(date=day.isoformat(), time=time_value.strftime("%H:%M"), party_size=int(party),
                            name=name.strip(), phone=phone.strip(), notes=notes.strip(), if_unavailable=policy)
    if edited != req:
        REQUEST_FILE.write_text(edited.model_dump_json(indent=2), encoding="utf-8")
        st.toast("Saved")

# ---------- 2. Agent ----------
PID_FILE = HERE / "agent.pid"  # survives page reloads, so the agent is never started twice


def agent_pid() -> int | None:
    """PID of a running agent.py, or None. Checked via tasklist, works after a page reload too."""
    if not PID_FILE.exists():
        return None
    pid = int(PID_FILE.read_text())
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    if str(pid) in out:
        return pid
    PID_FILE.unlink()
    return None


pid = agent_pid()

with st.container(border=True):
    st.subheader("2. Agent")
    if pid:
        st.success(f"Running · writing to @{MANAGER} · {llm.model_name()}")
        if st.button("Stop"):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            PID_FILE.unlink(missing_ok=True)
            db.stop_unfinished()
            st.rerun()
    else:
        st.info("Stopped")
        manager = st.text_input("Manager's Telegram", MANAGER, placeholder="@username").strip().lstrip("@")
        if manager != MANAGER:
            llm.save(manager=manager)
        choice = st.radio("Model", list(llm.PROVIDERS), format_func=llm.PROVIDERS.get, horizontal=True,
                          index=list(llm.PROVIDERS).index(llm.provider()))
        if choice == "ollama":
            models = llm.local_models() or [llm.DEFAULT_MODEL["ollama"]]
            current = llm.settings().get("ollama_model") or os.getenv("MODEL") or llm.DEFAULT_MODEL["ollama"]
            model = st.selectbox("Local model", models, index=models.index(current) if current in models else 0)
        else:
            model = st.text_input("Gemini model", llm.settings().get("gemini_model") or llm.DEFAULT_MODEL["gemini"])
            if not os.getenv("GEMINI_API_KEY"):
                st.warning("No GEMINI_API_KEY in .env")
        llm.save(llm=choice, **{f"{choice}_model": model})
        if st.button("Start", type="primary", disabled=not manager or (choice == "gemini" and not os.getenv("GEMINI_API_KEY"))):
            if LOG_FILE.exists():  # the previous conversation's file log goes to logs/
                (HERE / "logs").mkdir(exist_ok=True)
                LOG_FILE.rename(HERE / "logs" / f"decisions-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
            OWNER_FILE.unlink(missing_ok=True)
            db.stop_unfinished()  # e.g. the agent crashed or was killed outside the app
            env = {**os.environ, "OWNER_ANSWERS": "ui", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                   "MANAGER_USERNAME": manager}
            proc = subprocess.Popen(
                [sys.executable, str(HERE / "agent.py"), "telegram"], cwd=HERE, env=env,
                stdin=subprocess.DEVNULL, stdout=OUT_FILE.open("w", encoding="utf-8"), stderr=subprocess.STDOUT)
            PID_FILE.write_text(str(proc.pid))
            st.rerun()

    st.markdown("**Voice messages**")
    v1, v2 = st.columns([1, 2])
    start = v1.checkbox("Start with a voice message", value=voice.start_with_voice())
    then = v2.radio("Then reply", list(voice.THEN), format_func=voice.THEN.get, horizontal=True,
                    index=list(voice.THEN).index(voice.then()))
    engine = st.radio("Voice", list(voice.ENGINES), format_func=voice.ENGINES.get, horizontal=True,
                      index=list(voice.ENGINES).index(voice.engine()), disabled=not voice.CHATTERBOX_PYTHON)
    if (start, then, engine) != (voice.start_with_voice(), voice.then(), voice.engine()):
        llm.save(voice_start=start, voice_then=then, tts_engine=engine)
    if OUT_FILE.exists() and "Traceback" in OUT_FILE.read_text(encoding="utf-8", errors="replace"):
        with st.expander("⚠️ The agent crashed"):
            st.code(OUT_FILE.read_text(encoding="utf-8", errors="replace")[-2000:])


# ---------- shared helpers ----------
STATUS = {"confirmed": "✅ Confirmed", "no_booking": "👋 No booking", "in_progress": "⏳ In progress",
          "stopped": "⏹ Stopped"}


def status_label(status: str, card: dict) -> str:
    if status == "in_progress" and card.get("confirmed"):
        return "✅ Confirmed by the manager, finishing up"
    return STATUS.get(status, status)


def booking_day(text: str | None, started: str, request: dict) -> date:
    """The card's date is free text ("Tuesday 24 November", "today", ...): turn it into a real date."""
    t, start = (text or "").lower(), datetime.fromisoformat(started).date()
    m = re.search(r"(\d{1,2})\s+(" + "|".join(MONTHS) + ")", text or "", re.I)
    if m:
        d = date(start.year, MONTHS.index(m.group(2).capitalize()) + 1, int(m.group(1)))
        return d if d >= start - timedelta(days=1) else d.replace(year=start.year + 1)
    if "tomorrow" in t:
        return start + timedelta(days=1)
    if "today" in t or "tonight" in t:
        return start
    r = request.get("date") or "today"
    return start + timedelta(days=1) if r == "tomorrow" else start if r == "today" else date.fromisoformat(r)


def booking_line(b: dict, started: str, request: dict) -> str:
    d = booking_day(b.get("date"), started, request)
    return (f"**Restaurant:** {b.get('restaurant') or '—'} · **Date:** {d:%a %d %b %Y} · "
            f"**Time:** {b.get('time') or '—'} · **Guests:** {b.get('party_size') or '—'}")


def render_log(entries: list[dict], where: str) -> None:
    """The log as a chat: manager on the left, agent on the right, questions to the owner in the middle,
    and under each agent move what it understood and decided."""
    esc = lambda t: html.escape(str(t or ""))
    voice_note = lambda t: esc(t).replace("[voice message]", "🎤")
    parts = []
    for e in entries:
        t = esc((e.get("time") or "")[11:16])
        if e.get("manager_said"):
            parts.append(f'<div class="row l"><div class="b mgr"><div class="who">Manager · {t}</div>'
                         f'{voice_note(e["manager_said"])}</div></div>')
        if e.get("owner_said"):
            parts.append(f'<div class="row c"><div class="b own"><div class="who">Your answer</div>'
                         f'{esc(e["owner_said"])}</div></div>')
        parts.append(f'<div class="row r"><div class="note"><b>Understood:</b> {esc(e.get("understood"))}<br>'
                     f'<b>Decided:</b> {esc(e.get("decision"))}</div></div>')
        action = e.get("action")
        if action == "ask_owner":
            parts.append(f'<div class="row c"><div class="b ask"><div class="who">❓ The agent asks you · {t}</div>'
                         f'{esc(e.get("asked_owner"))}</div></div>')
        elif e.get("sent"):
            parts.append(f'<div class="row r"><div class="b bot"><div class="who">Agent · {t}</div>'
                         f'{esc(e["sent"])}</div></div>')
        else:
            parts.append('<div class="row r"><span class="chip">⏳ waiting for the manager</span></div>')
        if action == "finish":
            parts.append('<div class="row c"><span class="chip ok">✅ Booking confirmed</span></div>')
        elif action == "stop":
            parts.append('<div class="row c"><span class="chip">👋 Ended without a booking</span></div>')
    st.html("""<style>
        .chat{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:14px;line-height:1.45}
        .row{display:flex;margin:6px 0}.row.l{justify-content:flex-start}.row.r{justify-content:flex-end}
        .row.c{justify-content:center}
        .b{max-width:78%;padding:8px 12px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word}
        .who{font-size:11px;font-weight:600;opacity:.65;margin-bottom:2px}
        .mgr{background:#ECEAE4;color:#2C2C2A;border-bottom-left-radius:4px}
        .bot{background:#DDEFE5;color:#173D2B;border-bottom-right-radius:4px}
        .ask{background:#FFF1D6;color:#5A3A00;border:1px solid #F0C97A;max-width:88%}
        .own{background:#FFF8E8;color:#5A3A00;border:1px dashed #F0C97A;max-width:88%}
        .note{max-width:78%;font-size:12px;color:#6B6A65;background:#F7F6F2;border-left:3px solid #B9C9BF;
              padding:5px 9px;border-radius:6px}
        .note b{color:#4A5A50;font-weight:600}
        .chip{font-size:12px;padding:3px 10px;border-radius:999px;background:#EEEDE8;color:#5F5E5A}
        .chip.ok{background:#D5EDDF;color:#1E6B45;font-weight:600}
        </style><div class="chat">""" + "".join(parts) + "</div>")
    jsonl = "\n".join(json.dumps({k: v for k, v in e.items() if k not in ("id", "conversation_id")},
                                 ensure_ascii=False) for e in entries)
    st.download_button("Download log (JSONL)", jsonl.encode("utf-8"),
                       file_name=f"decisions-{entries[0]['conversation_id']}.jsonl",
                       key=f"dl-{where}-{entries[0]['conversation_id']}")


# ---------- 3–4. The agent's question, the current booking and conversation (refresh by themselves) ----------
@st.fragment(run_every="2s")
def live() -> None:
    if OWNER_FILE.exists():
        q = json.loads(OWNER_FILE.read_text(encoding="utf-8"))
        if not q.get("answer"):
            with st.container(border=True):
                st.subheader("❓ The agent asks you")
                st.write(q["question"])

                def reply(text: str) -> None:
                    OWNER_FILE.write_text(json.dumps({**q, "answer": text}), encoding="utf-8")

                options = q.get("options") or []
                if options:  # the manager offered several things: one button each (times, waiting list, none)
                    for i, (col, option) in enumerate(zip(st.columns(len(options)), options)):
                        if col.button(option, key=f"opt-{i}", type="primary" if i == 0 else "secondary",
                                      width="stretch"):
                            reply(option)
                            st.rerun()
                else:
                    b1, b2 = st.columns(2)
                    if b1.button("✅ Yes, book it", type="primary", width="stretch"):
                        reply("Yes, that works for me. Please go ahead and book it.")
                        st.rerun()
                    if b2.button("✖ No", width="stretch"):
                        reply("No, that doesn't work for me. Please thank them and say goodbye.")
                        st.rerun()
                with st.form("owner", clear_on_submit=True):
                    answer = st.text_input("Or answer in your own words")
                    if st.form_submit_button("Send") and answer:
                        reply(answer)
                        st.rerun()

    convs = db.conversations()
    current = convs[0] if convs else None
    entries = db.steps(current["id"]) if current else []

    with st.container(border=True):
        st.subheader("3. Current booking")
        if entries:
            b = entries[-1]["booking"]
            st.markdown(f"**Status:** {status_label(current['status'], b)}  \n"
                        + booking_line(b, current["started_at"], json.loads(current["request"])))
        else:
            st.caption("—")

    with st.container(border=True):
        st.subheader("4. Conversation")
        if entries:
            render_log(entries, "current")
        else:
            st.caption("—")


live()


# ---------- 5. History ----------
with st.container(border=True):
    st.subheader("5. History")
    convs = db.conversations()
    if not convs:
        st.caption("—")
    else:
        only_bookings = st.toggle("Confirmed only")
        shown = [c for c in convs if c["status"] == "confirmed"] if only_bookings else convs
        counts = db.step_counts()
        rows = []
        for c in shown:
            request = json.loads(c["request"])
            started = datetime.fromisoformat(c["started_at"])
            rows.append({
                "#": c["id"],
                "Started": f"{started:%d %b %H:%M}",
                "Status": STATUS.get(c["status"], c["status"]),
                "Restaurant": c["restaurant"] or "",
                "Date": f"{booking_day(c['booking_date'], c['started_at'], request):%a %d %b}",
                "Time": c["booking_time"] or request.get("time", ""),
                "Guests": str(c["party_size"] or request.get("party_size", "")),
                "Steps": counts.get(c["id"], 0),
            })
        picked = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun",
                              selection_mode="single-row", key="history")
        selected = picked.selection.rows
        if selected:
            conv = shown[selected[0]]
            st.markdown(f"**#{conv['id']}** · {STATUS.get(conv['status'], conv['status'])} · "
                        f"@{conv['manager'] or ''}")
            steps = db.steps(conv["id"])
            if steps:
                render_log(steps, "history")
            else:
                st.caption("No messages.")
        else:
            st.caption("Click a row to open its conversation.")
