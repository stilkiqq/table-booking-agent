"""Storage for conversations and bookings: one SQLite file, Python's built-in sqlite3, no server.

Tables:
    conversations  one row per booking attempt: what was requested, who the manager was, how it ended
    steps          one row per agent step: what the manager said, what the agent understood/decided/sent

A booking is a conversation with status 'confirmed'; its final details are in the booking_* columns.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_FILE = Path(__file__).parent / "bookings.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    manager       TEXT,
    request       TEXT NOT NULL,             -- the owner's request as JSON
    status        TEXT NOT NULL DEFAULT 'in_progress',  -- in_progress | confirmed | no_booking | stopped
    restaurant    TEXT,
    booking_date  TEXT,
    booking_time  TEXT,
    party_size    TEXT,
    booking_name  TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id),
    time             TEXT NOT NULL,
    manager_said     TEXT,
    understood       TEXT,
    decision         TEXT,
    action           TEXT,
    sent             TEXT,
    asked_owner      TEXT,
    owner_said       TEXT,                   -- the owner's answer that came before this step
    booking          TEXT                    -- the booking card after this step, as JSON
);
CREATE INDEX IF NOT EXISTS steps_by_conversation ON steps(conversation_id);
"""


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    if "owner_said" not in {r["name"] for r in con.execute("PRAGMA table_info(steps)")}:
        con.execute("ALTER TABLE steps ADD COLUMN owner_said TEXT")  # databases created before this column
    return con


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def start_conversation(request: dict, manager: str) -> int:
    with connect() as con:
        cur = con.execute("INSERT INTO conversations (started_at, manager, request) VALUES (?, ?, ?)",
                          (now(), manager, json.dumps(request, ensure_ascii=False)))
        return cur.lastrowid


def add_step(conversation_id: int, step: dict) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO steps (conversation_id, time, manager_said, understood, decision, action, sent, "
            "asked_owner, owner_said, booking) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, step["time"], step["manager_said"], step["understood"], step["decision"],
             step["action"], step["sent"], step["asked_owner"], step.get("owner_said", ""),
             json.dumps(step["booking"], ensure_ascii=False)))


def end_conversation(conversation_id: int, status: str, booking: dict) -> None:
    with connect() as con:
        con.execute(
            "UPDATE conversations SET ended_at = ?, status = ?, restaurant = ?, booking_date = ?, booking_time = ?, "
            "party_size = ?, booking_name = ? WHERE id = ?",
            (now(), status, booking.get("restaurant"), booking.get("date"), booking.get("time"),
             booking.get("party_size"), booking.get("name"), conversation_id))


def reopen_conversation(conversation_id: int) -> None:
    with connect() as con:
        con.execute("UPDATE conversations SET status = 'in_progress', ended_at = NULL WHERE id = ?", (conversation_id,))


def stop_unfinished() -> None:
    """Conversations still 'in_progress' when the agent is stopped or restarted will never finish: mark them."""
    with connect() as con:
        con.execute("UPDATE conversations SET status = 'stopped', ended_at = ? WHERE status = 'in_progress'", (now(),))


def conversations() -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM conversations ORDER BY id DESC")]


def step_counts() -> dict[int, int]:
    with connect() as con:
        return {r[0]: r[1] for r in con.execute("SELECT conversation_id, COUNT(*) FROM steps GROUP BY conversation_id")}


def steps(conversation_id: int) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM steps WHERE conversation_id = ? ORDER BY id", (conversation_id,))
        return [{**dict(r), "booking": json.loads(r["booking"] or "{}")} for r in rows]
