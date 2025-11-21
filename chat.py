  

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Dict, Any
import sqlite3
import uuid
import datetime
import os
import re
from openai import OpenAI

 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

 

app = FastAPI(title="Integrated Therapy Chatbot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 

def ask_llm(message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

 

class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CrisisSignal:
    def __init__(self, type: str, score: int, evidence: str):
        self.type = type
        self.score = score
        self.evidence = evidence

class CrisisDetectionResult:
    def __init__(self, is_crisis: bool, risk_level: RiskLevel, signals: List[CrisisSignal],
                 recommended_action: str, message: Optional[str] = None):
        self.is_crisis = is_crisis
        self.risk_level = risk_level
        self.signals = signals
        self.recommended_action = recommended_action
        self.message = message

class CrisisDetector:
    def __init__(self):
        # High / Medium / Low / Abuse patterns
        self.high_risk_patterns = [
            (r"\bi want to die\b", "suicide_intent", 5),
            (r"\bi am going to kill myself\b", "suicide_intent", 6),
            (r"\bi will kill myself\b", "suicide_intent", 6),
            (r"\bi plan to kill myself\b", "suicide_plan", 7),
            (r"\bi am going to (overdose|od)\b", "suicide_plan", 7),
            (r"\bi cut myself (today|now|again)\b", "self_harm_current", 6),
            (r"\bi am cutting myself\b", "self_harm_current", 7),
            (r"\bi just took (too many|a lot of) (pills|tablets)\b", "self_harm_current", 8),
            (r"\bi don't want to live anymore\b", "suicide_intent", 5),
            (r"\bi am going to kill (him|her|them)\b", "harm_others_intent", 7),
            (r"\bi will hurt (someone|him|her|them)\b", "harm_others_intent", 6),
        ]
        self.medium_risk_patterns = [
            (r"\bi want to hurt myself\b", "self_harm_ideation", 4),
            (r"\bi feel like killing myself\b", "suicide_ideation", 4),
            (r"\bi wish i were dead\b", "suicide_ideation", 4),
            (r"\beveryone would be better off without me\b", "suicide_ideation", 4),
            (r"\bi (sometimes|often) think about suicide\b", "suicide_ideation", 3),
            (r"\bi hate myself\b", "self_hatred", 2),
            (r"\bi am worthless\b", "self_hatred", 2),
        ]
        self.low_risk_patterns = [
            (r"\bi am extremely depressed\b", "depression", 2),
            (r"\bi feel hopeless\b", "hopelessness", 2),
            (r"\bi can't take this anymore\b", "overwhelmed", 2),
            (r"\bi feel like giving up\b", "overwhelmed", 2),
            (r"\bi am not okay\b", "distress", 1),
            (r"\bi am really struggling\b", "distress", 1),
        ]
        self.abuse_patterns = [
            (r"\bhe hits me\b", "physical_abuse", 4),
            (r"\bshe hits me\b", "physical_abuse", 4),
            (r"\bthey hit me\b", "physical_abuse", 4),
            (r"\bi am being abused\b", "abuse", 4),
            (r"\bi am afraid to go home\b", "safety_concern", 4),
        ]
        self.high_threshold = 6
        self.medium_threshold = 3

    def normalize(self, text: str):
        text = text.lower()
        return re.sub(r"\s+", " ", text).strip()

    def _match_patterns(self, text: str, patterns):
        signals = []
        for pattern, tag, score in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                signals.append(CrisisSignal(tag, score, match.group(0)))
        return signals

    def detect(self, message: str) -> CrisisDetectionResult:
        norm = self.normalize(message)
        signals = []
        signals.extend(self._match_patterns(norm, self.high_risk_patterns))
        signals.extend(self._match_patterns(norm, self.medium_risk_patterns))
        signals.extend(self._match_patterns(norm, self.low_risk_patterns))
        signals.extend(self._match_patterns(norm, self.abuse_patterns))

        if not signals:
            return CrisisDetectionResult(False, RiskLevel.NONE, [], "No crisis detected.", message)

        max_score = max(s.score for s in signals)
        total_score = sum(s.score for s in signals)

        if max_score >= self.high_threshold or total_score >= 10:
            level = RiskLevel.HIGH
            action = "High-risk crisis detected. Provide emergency help message."
        elif max_score >= self.medium_threshold or total_score >= 5:
            level = RiskLevel.MEDIUM
            action = "Medium-risk. Provide supportive and safety-check responses."
        else:
            level = RiskLevel.LOW
            action = "Low-risk distress. Provide supportive conversation."

        return CrisisDetectionResult(True, level, signals, action, message)

detector = CrisisDetector()

 

DB_FILE = "therapy.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # sessions
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            state TEXT,
            meta TEXT
        )
    """)
    # messages
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            step TEXT,
            message TEXT,
            created_at TEXT
        )
    """)
    # moods
    c.execute("""
        CREATE TABLE IF NOT EXISTS moods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value INTEGER,
            note TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def db_insert_session(session_id: str, state: str, meta: Optional[str] = None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO sessions(session_id, created_at, state, meta) VALUES (?, ?, ?, ?)",
              (session_id, datetime.datetime.utcnow().isoformat(), state, meta or ""))
    conn.commit()
    conn.close()

def db_update_session_state(session_id: str, state: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE sessions SET state = ? WHERE session_id = ?", (state, session_id))
    conn.commit()
    conn.close()

def db_insert_message(session_id: str, step: str, message: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages(session_id, step, message, created_at) VALUES (?, ?, ?, ?)",
              (session_id, step, message, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def db_save_mood(value: int, note: Optional[str]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO moods(value, note, created_at) VALUES (?, ?, ?)",
              (value, note or "", datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def db_get_recent_moods(limit: int = 10):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT value, note, created_at FROM moods ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"value": r[0], "note": r[1], "created_at": r[2]} for r in rows]

def db_get_session_state(session_id: str) -> Optional[str]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT state FROM sessions WHERE session_id = ?", (session_id,))
    r = c.fetchone()
    conn.close()
    return r[0] if r else None

 
class CBTRespondRequest(BaseModel):
    session_id: str
    message: str

class CBTRespondResponse(BaseModel):
    next_step: str
    prompt: str
    finished: Optional[bool] = False
    reframe: Optional[str] = None

CBT_STEPS = [
    ("step1", "What's the negative thought you have?"),
    ("step2", "Evidence that supports the thought?"),
    ("step3", "Evidence against the thought?"),
    ("step4", "AI will reframe into balanced alternative."),
]

def simple_reframe(negative_thought: str, evidence: str, counter_evidence: str) -> str:
    return f"Balanced thought: While '{negative_thought}', evidence shows '{counter_evidence}'."

 
@app.post("/chat")
async def chat_endpoint(request: dict):
    user_msg = request.get("message", "")
    # Crisis check first
    crisis_result = detector.detect(user_msg)
    if crisis_result.is_crisis and crisis_result.risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]:
        return {
            "response": "⚠️ Crisis detected! Please contact emergency services or a trusted person immediately.",
            "risk_level": crisis_result.risk_level.value,
            "signals": [(s.type, s.evidence) for s in crisis_result.signals]
        }
    # Otherwise, normal LLM reply
    reply = ask_llm(user_msg)
    return {"response": reply}

@app.post("/cbt/respond", response_model=CBTRespondResponse)
async def cbt_respond(req: CBTRespondRequest):
    state = db_get_session_state(req.session_id)
    if not state:
        db_insert_session(req.session_id, CBT_STEPS[0][0])
        state = CBT_STEPS[0][0]
    db_insert_message(req.session_id, state, req.message)
    # next step
    idx = next((i for i, s in enumerate(CBT_STEPS) if s[0]==state), 0)
    if idx < len(CBT_STEPS)-1:
        next_state = CBT_STEPS[idx+1][0]
        db_update_session_state(req.session_id, next_state)
        prompt = CBT_STEPS[idx+1][1]
        return CBTRespondResponse(next_step=next_state, prompt=prompt, finished=False)
    else:
        # final step
        reframe = simple_reframe("", "", "")  # For simplicity
        db_update_session_state(req.session_id, "finished")
        return CBTRespondResponse(next_step="finished", prompt="Session complete", finished=True, reframe=reframe)

@app.post("/mood")
async def post_mood(mood: Dict[str, Any]):
    db_save_mood(mood.get("value",0), mood.get("note",""))
    return {"ok": True}

@app.get("/mood/recent")
async def get_recent_moods(limit: int = 10):
    rows = db_get_recent_moods(limit)
    return {"moods": rows}

@app.get("/")
async def root():
    return {"msg": "Integrated Therapy Chatbot running"}

 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
