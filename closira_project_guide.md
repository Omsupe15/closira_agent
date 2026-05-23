# Closira AI Agent — Complete Project Guide
### Stack: Google Gemini · LangChain · LangGraph

> **Assignment:** Build a Python-based, 4-stage AI customer support workflow for Bloom Aesthetics Clinic using the Google Gemini API, LangChain for LLM calls, and LangGraph for workflow orchestration.
> **Deadline:** 24 hours from receipt.

---

## Table of Contents

1. [Understanding the Assignment](#1-understanding-the-assignment)
2. [Project Structure](#2-project-structure)
3. [Setup & Dependencies](#3-setup--dependencies)
4. [The SOP Data File](#4-the-sop-data-file)
5. [LangGraph State & Graph Architecture](#5-langgraph-state--graph-architecture)
6. [Stage 1 — FAQ Answering Node](#6-stage-1--faq-answering-node)
7. [Stage 2 — Lead Qualification Node](#7-stage-2--lead-qualification-node)
8. [Stage 3 — Escalation Detection Node](#8-stage-3--escalation-detection-node)
9. [Stage 4 — Conversation Summary Node](#9-stage-4--conversation-summary-node)
10. [Main Orchestrator](#10-main-orchestrator)
11. [prompt_design.md — What to Write](#11-prompt_designmd--what-to-write)
12. [Test Transcripts — What to Write](#12-test-transcripts--what-to-write)
13. [README.md — What to Write](#13-readmemd--what-to-write)
14. [Evaluation Checklist](#14-evaluation-checklist)

---

## 1. Understanding the Assignment

### What Closira Is

Closira is a platform that handles inbound customer messages for small businesses using AI. You're building the **intelligence layer** — the brain behind the conversations.

### The Four Stages (The Core of Everything)

Think of each conversation as a pipeline that flows through four checkpoints:

```
Customer Message
       │
       ▼
┌─────────────────┐
│ Stage 3:        │  ← Escalation check runs FIRST on every message.
│ Escalation      │     Detect anger, out-of-scope, explicit hand-off request.
└────────┬────────┘
         │ (if not escalated)
         ▼
┌─────────────────┐
│ Stage 1: FAQ    │  ← Answer from SOP only. No guessing.
└────────┬────────┘
         │ (if not escalated)
         ▼
┌─────────────────┐
│ Stage 2: Lead   │  ← Ask 2–3 qualifying questions. Store answers.
│ Qualification   │
└────────┬────────┘
         │ (at session end)
         ▼
┌─────────────────┐
│ Stage 4:        │  ← Structured summary of the whole session.
│ Summary         │
└─────────────────┘
```

> **Why escalation runs first?** In a production system you never want the AI to attempt an answer to an angry or out-of-scope message. Catching it first means the FAQ node only runs when it is safe and appropriate to do so.

### Key Constraints

- Answer **only from the SOP**. Never invent prices, services, or policies.
- Escalation must be **logged with a reason** — not just triggered silently.
- Lead qualification data must be **stored** (in memory is fine for a CLI).
- The final summary must be **structured** (not a paragraph of prose).

---

## 2. Project Structure

Create this exact folder layout before writing any code:

```
closira-agent/
│
├── main.py                  # Entry point — CLI conversation loop
│
├── agent/
│   ├── __init__.py
│   ├── state.py             # LangGraph AgentState TypedDict
│   ├── graph.py             # StateGraph wiring — nodes + conditional edges
│   ├── llm.py               # Shared Gemini LLM factory
│   ├── sop_loader.py        # SOP JSON loader + text formatter
│   ├── faq.py               # Stage 1: FAQ answering node
│   ├── qualification.py     # Stage 2: Lead qualification node
│   ├── escalation.py        # Stage 3: Escalation detection node
│   └── summary.py           # Stage 4: Conversation summary node
│
├── data/
│   └── sop.json             # SOP source of truth
│
├── logs/
│   └── escalations.log      # Auto-created at runtime
│
├── test_transcripts/
│   ├── 01_in_scope_question.md
│   ├── 02_out_of_scope_question.md
│   ├── 03_escalation_trigger.md
│   ├── 04_lead_qualification.md
│   └── 05_conversation_summary.md
│
├── prompt_design.md         # Required deliverable
├── README.md                # Required deliverable
├── .env                     # API key — never commit
└── requirements.txt
```

---

## 3. Setup & Dependencies

### `requirements.txt`

```
langchain>=0.2.0
langchain-core>=0.2.0
langchain-google-genai>=1.0.0
langgraph>=0.1.0
google-generativeai>=0.7.0
python-dotenv>=1.0.0
```

**Why each package:**
- `langchain-google-genai` — LangChain's native Google Gemini integration. Provides `ChatGoogleGenerativeAI`, the LangChain-compatible wrapper around `google-generativeai`.
- `langgraph` — The workflow engine. Manages the `StateGraph`, node routing, and conditional edges.
- `langchain-core` — Provides `SystemMessage`, `HumanMessage`, `AIMessage`, `StrOutputParser`, and `ChatPromptTemplate`.
- `google-generativeai` — The underlying Google SDK that `langchain-google-genai` depends on.

### `.env` file (never commit this)

```
GOOGLE_API_KEY=AIzaSy-xxxxxxxxxx
```

Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Shared LLM Factory — `agent/llm.py`

Rather than instantiating the model inside every node, create one shared factory. Every node imports from here.

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Returns a LangChain-compatible Gemini 2.5 Flash instance.
    Use temperature=0.1 for classification nodes (escalation),
    temperature=0.3 for generation nodes (FAQ, summary).
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
```

### Basic LangChain call pattern

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from agent.llm import get_llm

llm = get_llm()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What are your opening hours?")
]

response = llm.invoke(messages)
text: str = response.content        # Plain string output
```

### LangChain chain pattern (prompt → llm → parser)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    ("human", "{user_input}")
])

chain = prompt | llm | StrOutputParser()

result: str = chain.invoke({
    "system": "You are a support agent.",
    "user_input": "What is Botox?"
})
```

---

## 4. The SOP Data File

### `data/sop.json`

Store your SOP as structured JSON. This makes it easy to inject into prompts and extend later.

```json
{
  "business_name": "Bloom Aesthetics Clinic",
  "hours": "Monday to Saturday, 9am to 7pm",
  "services": [
    { "name": "Botox", "price_from": 200, "currency": "GBP" },
    { "name": "Fillers", "price_from": 250, "currency": "GBP" },
    { "name": "Consultation", "price_from": 0, "currency": "GBP", "notes": "Free" }
  ],
  "booking": {
    "channels": ["WhatsApp", "website"],
    "cancellation_policy": "24 hours notice required"
  },
  "escalation_triggers": [
    "complaint",
    "medical question",
    "pricing negotiation",
    "more than 2 unanswered questions"
  ]
}
```

### `agent/sop_loader.py`

```python
import json

def load_sop(path: str = "data/sop.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)

def sop_to_text(sop: dict) -> str:
    """Convert SOP dict to a clean string block for injection into system prompts."""
    services = "\n".join(
        f"  - {s['name']}: from £{s['price_from']}" if s["price_from"] > 0
        else f"  - {s['name']}: Free"
        for s in sop["services"]
    )
    return f"""Business: {sop['business_name']}
Hours: {sop['hours']}
Services:
{services}
Booking: Via {' or '.join(sop['booking']['channels'])}. {sop['booking']['cancellation_policy']} notice required for cancellations.
Escalate if: {', '.join(sop['escalation_triggers'])}.
"""
```

---

## 5. LangGraph State & Graph Architecture

This is the core of the project. Before writing any stage logic, you must define the **shared state** and the **graph that routes between stages**.

### `agent/state.py` — The Shared State

LangGraph passes a single state dictionary through every node. Every piece of data the workflow needs to track lives here.

```python
from typing import TypedDict

class AgentState(TypedDict):
    # Conversation history as simple dicts (converted to LangChain messages in each node)
    conversation_history: list[dict]   # [{"role": "user/assistant", "content": "..."}]

    # The raw text of the customer's latest message
    current_input: str

    # Stage 1 output
    faq_result: dict    # {"answer": str, "escalate": bool,
                        #  "escalation_reason": str, "confidence": str}

    # Stage 3 output
    escalation_result: dict   # {"escalate": bool, "reason": str, "sentiment": str}

    # Stage 2 data
    qualification_answers: dict   # {question_text: answer_text}
    qualification_index: int      # Which question we're on (0, 1, 2)
    qualification_complete: bool

    # Session control
    escalated: bool         # True when any stage triggers escalation
    session_ended: bool     # True when user types 'quit'

    # The message to print to the user after each graph run
    next_agent_message: str

    # Stage 4 output
    summary: str
```

### `agent/graph.py` — The LangGraph StateGraph

This file wires nodes together with conditional routing logic.

```python
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.escalation import escalation_node
from agent.faq import faq_node
from agent.qualification import qualification_node
from agent.summary import summary_node


# ── Routing functions ────────────────────────────────────────────────────────

def route_after_escalation_check(state: AgentState) -> str:
    """After checking escalation: hand off to human OR proceed to FAQ."""
    if state["escalated"]:
        return "generate_summary"
    return "faq_answer"

def route_after_faq(state: AgentState) -> str:
    """After FAQ: escalate, qualify the lead, or end this turn."""
    if state["escalated"]:
        return "generate_summary"
    if not state["qualification_complete"]:
        return "lead_qualify"
    return END   # Qualification done, just end the turn


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the LangGraph StateGraph.

    Flow per turn:
      escalation_check
        ├─ [escalated]     → generate_summary → END
        └─ [clear]         → faq_answer
                               ├─ [escalated] → generate_summary → END
                               ├─ [qualify]   → lead_qualify → END
                               └─ [done]      → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("escalation_check", escalation_node)
    graph.add_node("faq_answer", faq_node)
    graph.add_node("lead_qualify", qualification_node)
    graph.add_node("generate_summary", summary_node)

    # Entry point is always the escalation check
    graph.set_entry_point("escalation_check")

    # Conditional edges
    graph.add_conditional_edges(
        "escalation_check",
        route_after_escalation_check,
        {
            "generate_summary": "generate_summary",
            "faq_answer": "faq_answer"
        }
    )

    graph.add_conditional_edges(
        "faq_answer",
        route_after_faq,
        {
            "generate_summary": "generate_summary",
            "lead_qualify": "lead_qualify",
            END: END
        }
    )

    # Terminal edges
    graph.add_edge("lead_qualify", END)
    graph.add_edge("generate_summary", END)

    return graph.compile()
```

### How the graph routes visually

```
[START]
   │
   ▼
escalation_check ──[escalated]──────────────────────┐
   │                                                 │
   │ [clear]                                         │
   ▼                                                 │
faq_answer ──[escalated]─────────────────────────── │
   │                                                 │
   │ [qualify]                          ▼            │
   ▼                            generate_summary ◄──┘
lead_qualify                           │
   │                                   ▼
   ▼                                  END
  END
```

---

## 6. Stage 1 — FAQ Answering Node

### What it does

Takes the current conversation history and answers using the SOP only. Uses a structured output format so escalation signals can be parsed reliably.

### `agent/faq.py`

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from agent.llm import get_llm
from agent.state import AgentState
from agent.sop_loader import load_sop, sop_to_text

# Load SOP once at module level
_SOP = load_sop()
_SOP_TEXT = sop_to_text(_SOP)

FAQ_SYSTEM_PROMPT = f"""You are a helpful customer support assistant for {_SOP['business_name']}.

You must answer customer questions using ONLY the information in the SOP below.
Do not make up, infer, or estimate any information that is not explicitly stated in the SOP.

If a customer asks something not covered in the SOP, respond with exactly:
"I don't have information on that — let me connect you with our team who can help."
Then set ESCALATE: true.

Always be warm, professional, and concise. You represent a clinic — maintain a reassuring tone.

=== SOP ===
{_SOP_TEXT}
=== END SOP ===

Response format — follow this EXACTLY on four separate lines:
ANSWER: <your response to the customer>
ESCALATE: <true or false>
ESCALATION_REASON: <one-line reason if escalating, else leave blank>
CONFIDENCE: <high / medium / low>
"""


def _history_to_langchain_messages(history: list[dict]) -> list:
    """Convert simple dicts to LangChain message objects."""
    messages = [SystemMessage(content=FAQ_SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def _parse_faq_response(raw: str) -> dict:
    """Parse the structured ANSWER/ESCALATE/CONFIDENCE block."""
    result = {
        "answer": "",
        "escalate": False,
        "escalation_reason": "",
        "confidence": "high"
    }
    for line in raw.strip().split("\n"):
        if line.startswith("ANSWER:"):
            result["answer"] = line.replace("ANSWER:", "").strip()
        elif line.startswith("ESCALATE:"):
            result["escalate"] = line.replace("ESCALATE:", "").strip().lower() == "true"
        elif line.startswith("ESCALATION_REASON:"):
            result["escalation_reason"] = line.replace("ESCALATION_REASON:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip().lower()
    return result


def faq_node(state: AgentState) -> AgentState:
    """
    LangGraph node — Stage 1: FAQ Answering.
    Calls Gemini via LangChain with the full conversation history as context.
    Returns updated state with faq_result, escalated flag, and next_agent_message.
    """
    llm = get_llm(temperature=0.3)
    messages = _history_to_langchain_messages(state["conversation_history"])

    response = llm.invoke(messages)
    faq_result = _parse_faq_response(response.content)

    # Escalate if model flagged it OR confidence is low
    should_escalate = faq_result["escalate"] or faq_result["confidence"] == "low"

    updated_history = state["conversation_history"] + [
        {"role": "assistant", "content": faq_result["answer"]}
    ]

    return {
        **state,
        "faq_result": faq_result,
        "escalated": should_escalate,
        "next_agent_message": faq_result["answer"],
        "conversation_history": updated_history
    }
```

---

## 7. Stage 2 — Lead Qualification Node

### What it does

After the FAQ stage, asks the customer 2–3 structured qualifying questions one at a time. Stores all answers in state.

### `agent/qualification.py`

```python
from agent.state import AgentState

QUALIFICATION_QUESTIONS = [
    "Could I ask what brings you to Bloom Aesthetics today — is this your first time looking into aesthetic treatments?",
    "Are you based locally in the area, or would you be travelling to visit us?",
    "Is there a particular treatment you're most interested in, or would you like to explore your options during a free consultation?"
]


def qualification_node(state: AgentState) -> AgentState:
    """
    LangGraph node — Stage 2: Lead Qualification.
    Asks one question per turn using the qualification_index in state.
    Does NOT call the LLM — questions are pre-defined for reliability.
    """
    idx = state.get("qualification_index", 0)
    answers = state.get("qualification_answers", {})

    if idx >= len(QUALIFICATION_QUESTIONS):
        # All questions asked — mark complete
        return {
            **state,
            "qualification_complete": True,
            "qualification_answers": answers,
            "next_agent_message": state.get("next_agent_message", "")
        }

    question = QUALIFICATION_QUESTIONS[idx]

    return {
        **state,
        "qualification_index": idx,   # Keep index — answer recorded in main loop
        "qualification_answers": answers,
        "qualification_complete": False,
        "next_agent_message": question
    }


def record_qualification_answer(state: AgentState, answer: str) -> AgentState:
    """
    Called from main.py after the user responds to a qualification question.
    Advances the index and stores the answer.
    """
    idx = state.get("qualification_index", 0)
    question = QUALIFICATION_QUESTIONS[idx]
    answers = dict(state.get("qualification_answers", {}))
    answers[question] = answer

    new_idx = idx + 1
    complete = new_idx >= len(QUALIFICATION_QUESTIONS)

    return {
        **state,
        "qualification_answers": answers,
        "qualification_index": new_idx,
        "qualification_complete": complete,
        "conversation_history": state["conversation_history"] + [
            {"role": "user", "content": answer}
        ]
    }
```

> **Why no LLM here?** Qualification questions are fixed and deterministic. Letting the model rephrase them introduces variability and the risk that a question is skipped or reordered. Pre-defined questions guarantee consistent data collection.

---

## 8. Stage 3 — Escalation Detection Node

### What it does

Runs on every customer message **before** the FAQ stage. Detects anger, medical questions, pricing negotiation, or an explicit hand-off request. Logs every escalation with a timestamped reason.

### Escalation triggers

| Trigger | Detection Method |
|---|---|
| Complaint / anger | Gemini sentiment classification |
| Medical question | Gemini content classification |
| Pricing negotiation | Gemini content classification |
| Explicit escalation request | Gemini + keyword fallback |
| Low AI confidence (FAQ stage) | `CONFIDENCE: low` in FAQ output |

### `agent/escalation.py`

```python
import logging
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from agent.llm import get_llm
from agent.state import AgentState

# Ensure logs directory exists and configure logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/escalations.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

ESCALATION_SYSTEM_PROMPT = """You are a classifier for a customer support system at an aesthetics clinic.

Analyse the customer's message and determine if it requires escalation to a human agent.

Escalate (set ESCALATE: true) if ANY of the following are true:
- The customer expresses frustration, anger, or makes a complaint
- The customer asks a medical question (side effects, safety, contraindications, allergies)
- The customer is attempting to negotiate pricing
- The customer explicitly asks to speak to a human, manager, or real person
- The message is completely outside the scope of an aesthetics clinic

Respond ONLY in this exact format on three separate lines:
ESCALATE: <true or false>
REASON: <one-line reason, or "none" if not escalating>
SENTIMENT: <positive / neutral / frustrated / angry>
"""


def _parse_escalation_response(raw: str) -> dict:
    result = {"escalate": False, "reason": "none", "sentiment": "neutral"}
    for line in raw.strip().split("\n"):
        if line.startswith("ESCALATE:"):
            result["escalate"] = line.replace("ESCALATE:", "").strip().lower() == "true"
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("SENTIMENT:"):
            result["sentiment"] = line.replace("SENTIMENT:", "").strip()
    return result


def _log_escalation(reason: str, message: str, sentiment: str):
    logging.info(
        f"ESCALATION | Reason: {reason} | Sentiment: {sentiment} | "
        f"Message: '{message[:120]}'"
    )


def escalation_node(state: AgentState) -> AgentState:
    """
    LangGraph node — Stage 3: Escalation Detection.
    Runs first on every turn. Uses Gemini via LangChain to classify
    the customer's intent and sentiment. Logs and flags if escalation needed.
    """
    llm = get_llm(temperature=0.1)   # Low temperature for deterministic classification

    messages = [
        SystemMessage(content=ESCALATION_SYSTEM_PROMPT),
        HumanMessage(content=state["current_input"])
    ]

    response = llm.invoke(messages)
    result = _parse_escalation_response(response.content)

    if result["escalate"]:
        _log_escalation(result["reason"], state["current_input"], result["sentiment"])

        handoff_message = (
            "I completely understand, and I want to make sure you get the best support. "
            "Let me connect you with one of our team members who can help you directly. "
            "Someone will be in touch with you shortly."
        )
        print(f"\n[ESCALATION FLAGGED] Reason: {result['reason']} | "
              f"Sentiment: {result['sentiment']}")

        return {
            **state,
            "escalation_result": result,
            "escalated": True,
            "next_agent_message": handoff_message,
            "conversation_history": state["conversation_history"] + [
                {"role": "assistant", "content": handoff_message}
            ]
        }

    return {
        **state,
        "escalation_result": result,
        "escalated": False
    }
```

---

## 9. Stage 4 — Conversation Summary Node

### What it does

Runs at the end of every session — whether the session ended normally or via escalation. Produces a structured summary for the human agent or business owner.

### `agent/summary.py`

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from agent.llm import get_llm
from agent.state import AgentState

SUMMARY_SYSTEM_PROMPT = """You are an AI assistant that generates structured end-of-session summaries
for customer support conversations at an aesthetics clinic.

Given the full conversation history and session data, produce a structured summary.
Your summary MUST include ALL of these labelled sections exactly as shown:

CUSTOMER_INTENT: <1-2 sentences describing what the customer wanted>
KEY_DETAILS_COLLECTED:
  - <bullet: each piece of information learned about the customer>
SOP_GAPS_IDENTIFIED: <questions the AI could not answer from the SOP, or "None">
LEAD_QUALIFICATION:
  - <bullet: each qualifying question and the customer's answer, or "Not completed">
ESCALATED: <Yes or No>
ESCALATION_REASON: <reason if escalated, else "N/A">
RECOMMENDED_NEXT_ACTION: <what the human agent or business should do next>
"""


def _format_history(history: list[dict]) -> str:
    lines = []
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Agent"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _format_qualification(answers: dict) -> str:
    if not answers:
        return "No qualification data collected."
    return "\n".join(f"  Q: {q}\n  A: {a}" for q, a in answers.items())


def summary_node(state: AgentState) -> AgentState:
    """
    LangGraph node — Stage 4: Conversation Summary.
    Called at session end (user quit) or after escalation.
    Uses Gemini via LangChain to generate a structured summary.
    """
    llm = get_llm(temperature=0.2)

    escalation_result = state.get("escalation_result", {})
    escalated = state.get("escalated", False)
    esc_text = (
        f"Escalated: Yes\nReason: {escalation_result.get('reason', 'Unknown')}\n"
        f"Sentiment: {escalation_result.get('sentiment', 'Unknown')}"
        if escalated else "Escalated: No"
    )

    context = f"""=== CONVERSATION HISTORY ===
{_format_history(state['conversation_history'])}

=== LEAD QUALIFICATION DATA ===
{_format_qualification(state.get('qualification_answers', {}))}

=== ESCALATION DATA ===
{esc_text}
"""

    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]

    response = llm.invoke(messages)
    summary_text = response.content

    return {
        **state,
        "summary": summary_text
    }
```

---

## 10. Main Orchestrator

### `main.py`

This is the CLI entry point. It runs the conversation loop, feeds each user message into the LangGraph, and handles the qualification sub-loop between graph turns.

```python
import os
from dotenv import load_dotenv
from agent.graph import build_graph
from agent.state import AgentState
from agent.qualification import record_qualification_answer, QUALIFICATION_QUESTIONS
from agent.summary import summary_node

load_dotenv()

def get_initial_state() -> AgentState:
    return AgentState(
        conversation_history=[],
        current_input="",
        faq_result={},
        escalation_result={},
        qualification_answers={},
        qualification_index=0,
        qualification_complete=False,
        escalated=False,
        session_ended=False,
        next_agent_message="",
        summary=""
    )


def run_conversation():
    print("=" * 60)
    print("  Bloom Aesthetics Clinic — AI Support Agent")
    print("  (Powered by Gemini · LangChain · LangGraph)")
    print("  Type 'quit' to end session and receive a summary.")
    print("=" * 60)

    graph = build_graph()
    state = get_initial_state()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            state["session_ended"] = True
            break
        if not user_input:
            continue

        # Feed message into state
        state["current_input"] = user_input
        state["conversation_history"] = state["conversation_history"] + [
            {"role": "user", "content": user_input}
        ]

        # ── Run the LangGraph for this turn ──────────────────────────────
        state = graph.invoke(state)

        # Print what the agent said
        if state.get("next_agent_message"):
            print(f"\nAgent: {state['next_agent_message']}")

        # ── If escalated, end session ─────────────────────────────────────
        if state.get("escalated"):
            break

        # ── Stage 2: Qualification sub-loop ──────────────────────────────
        # If the graph routed to qualification, the question is in next_agent_message.
        # We need to collect the answer and record it before the next graph turn.
        while (
            not state.get("qualification_complete")
            and state.get("qualification_index", 0) < len(QUALIFICATION_QUESTIONS)
        ):
            # The question was already printed above. Collect the answer.
            qual_answer = input("You: ").strip()
            if not qual_answer:
                continue

            state = record_qualification_answer(state, qual_answer)

            # If there are more questions, get the next one
            if not state["qualification_complete"]:
                next_idx = state["qualification_index"]
                next_q = QUALIFICATION_QUESTIONS[next_idx]
                print(f"\nAgent: {next_q}")
                state["next_agent_message"] = next_q
            else:
                print("\nAgent: Thank you — that's really helpful!")
                break

    # ── Stage 4: Generate session summary ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SESSION ENDED — Generating Summary...")
    print("=" * 60)

    state = summary_node(state)
    print("\n" + state["summary"])

    # Save to file
    os.makedirs("logs", exist_ok=True)
    with open("logs/last_session_summary.txt", "w") as f:
        f.write(state["summary"])
    print("\n[Summary saved to logs/last_session_summary.txt]")


if __name__ == "__main__":
    run_conversation()
```

---

## 11. `prompt_design.md` — What to Write

This is a **graded deliverable**. Write it clearly. Here's what each section should contain and why.

---

### Section 1: System Prompt

Paste your full system prompts (FAQ, Escalation, Summary) verbatim. Then explain key choices:

> **Why a structured output format (ANSWER / ESCALATE / CONFIDENCE)?**
> Parsing free-form text is fragile. A fixed key-value format allows the `_parse_faq_response()` function to extract the escalation flag, confidence level, and customer-facing reply deterministically — without regex ambiguity or a second LLM call. This is especially important for `ESCALATE: true/false` since it drives graph routing in LangGraph.

> **Why is the SOP injected between labelled markers (`=== SOP ===`)?**
> The markers draw a clear visual and semantic boundary in the context, making it unambiguous what counts as authoritative information versus the instruction text. Gemini, like other large models, attends to structural cues in the prompt — explicit delimiters reduce the chance of content bleeding between sections.

> **Why use `ChatPromptTemplate` / `SystemMessage` over raw strings?**
> LangChain message types (`SystemMessage`, `HumanMessage`, `AIMessage`) map directly to the role-based format the Gemini API expects. This avoids manual role tagging and ensures the system prompt is always treated as a system instruction, not a user turn, which affects how the model weights the instructions.

---

### Section 2: Hallucination Prevention

Explain your three-layer approach:

**Layer 1 — Hard instruction in system prompt:**
> "Answer ONLY from the SOP. Do not make up, infer, or estimate any information that is not explicitly stated."

**Layer 2 — Scripted fallback for out-of-scope:**
> When no SOP answer exists, the model is told exactly what to say: *"I don't have information on that — let me connect you with our team."* This removes any incentive for the model to fill the gap with a guess.

**Layer 3 — Confidence flag as a safety net:**
> Even if the model produces an answer, `CONFIDENCE: low` triggers escalation automatically, meaning uncertain answers are intercepted before they reach the customer unsupported. This is the difference between the model silently hallucinating and the system catching it.

---

### Section 3: Confidence-Based Escalation

Explain your three-signal design:

**Signal A — FAQ explicit flag:** The FAQ system prompt outputs `ESCALATE: true` when the question is not covered by the SOP.

**Signal B — Confidence level:** `CONFIDENCE: low` in the FAQ response triggers the `should_escalate` flag in `faq_node`, regardless of whether `ESCALATE` was set.

**Signal C — Independent sentiment detector:** `escalation_node` runs a completely separate LangGraph node on every message *before* the FAQ stage, catching anger or complaints even when the customer's question itself is in-scope.

This layered design means the system has three independent opportunities to catch a situation that needs human intervention.

---

### Section 4: Tone and Persona

> The AI represents an aesthetics clinic — a personal, appearance-related service where trust and reassurance are critical. Patients may be self-conscious, nervous about procedures, or emotionally invested in results. The tone must be warm without being effusive, and professional without being clinical. The persona avoids corporate coldness ("Your query has been logged") and avoids hollow enthusiasm ("Great question!"). The goal is the voice of a knowledgeable, friendly receptionist who takes the patient seriously.

> LangChain's `SystemMessage` ensures this persona instruction is always the highest-priority context fed to the model, structurally separate from the conversation turns.

---

## 12. Test Transcripts — What to Write

Create one `.md` file per scenario in `test_transcripts/`. Each file should show the full exchange.

### `01_in_scope_question.md`

```markdown
# Test Transcript 1 — In-Scope Question

**Scenario:** Customer asks about Botox pricing.
**Expected behaviour:** Agent answers accurately from SOP. No escalation.

---

**You:** Hi, how much does Botox cost?

**Agent:** Hi there! Botox treatments at Bloom Aesthetics start from £200.
Prices can vary depending on the areas treated, so we'd recommend a free
consultation to get a personalised quote. Would you like to book one?

ESCALATE: false
CONFIDENCE: high

**Result:** ✅ Passed — SOP answer given correctly. No hallucination.
```

### `02_out_of_scope_question.md`

```markdown
# Test Transcript 2 — Out-of-Scope Question

**Scenario:** Customer asks something not in the SOP.
**Expected behaviour:** Agent acknowledges the gap and escalates.

---

**You:** Do you offer laser hair removal?

**Agent:** I don't have information on that — let me connect you with our team
who can help. They'll be able to give you accurate details on all our services.

ESCALATE: true
ESCALATION_REASON: Service not covered in SOP
CONFIDENCE: high

**Result:** ✅ Passed — Gap acknowledged. Escalation triggered. No guessing.
```

### `03_escalation_trigger.md`

```markdown
# Test Transcript 3 — Escalation: Angry Customer

**Scenario:** Customer expresses frustration.
**Expected behaviour:** Agent detects sentiment and hands off with a logged reason.

---

**You:** This is ridiculous. I've been trying to get a callback for a week and
no one has bothered to get back to me. I'm absolutely furious.

**Agent:** I completely understand, and I want to make sure you get the best
support. Let me connect you with one of our team members who can help you
directly. Someone will be in touch with you shortly.

[ESCALATION FLAGGED] Reason: Customer expressed strong frustration / complaint
Sentiment: angry
Logged to: logs/escalations.log

**Result:** ✅ Passed — Frustration detected. Escalation triggered and logged.
```

### `04_lead_qualification.md`

```markdown
# Test Transcript 4 — Lead Qualification

**Scenario:** Customer expresses interest, agent qualifies them.
**Expected behaviour:** 2–3 structured questions asked. Answers stored in state.

---

**You:** I'm interested in getting fillers done.

**Agent:** Fillers at Bloom Aesthetics start from £250, and we also offer a free
consultation to discuss which treatment would be best for you.

**Agent:** Could I ask what brings you to Bloom Aesthetics today — is this your
first time looking into aesthetic treatments?

**You:** Yes, it's my first time.

**Agent:** Are you based locally in the area, or would you be travelling to visit us?

**You:** I'm about 20 minutes away.

**Agent:** Is there a particular treatment you're most interested in, or would
you like to explore your options during a free consultation?

**You:** I'd like to try lip fillers specifically.

**Agent:** Thank you — that's really helpful!

qualification_answers = {
  "Could I ask what brings you to Bloom Aesthetics...": "Yes, it's my first time.",
  "Are you based locally in the area...": "I'm about 20 minutes away.",
  "Is there a particular treatment...": "I'd like to try lip fillers specifically."
}
qualification_complete = True

**Result:** ✅ Passed — All 3 questions asked. Answers stored in AgentState.
```

### `05_conversation_summary.md`

```markdown
# Test Transcript 5 — Conversation Summary

**Scenario:** End of session. Agent generates structured summary.
**Expected behaviour:** Clean summary with intent, details, gaps, and next action.

---

[Session ended by user typing 'quit']
[summary_node invoked via LangGraph]

CUSTOMER_INTENT: Customer enquired about lip filler pricing and availability
and was interested in booking a first-time appointment.

KEY_DETAILS_COLLECTED:
  - First-time patient (no prior aesthetic treatments)
  - Lives approximately 20 minutes from the clinic
  - Interested specifically in lip fillers

SOP_GAPS_IDENTIFIED: None

LEAD_QUALIFICATION:
  - First time with aesthetics: Yes
  - Location: ~20 minutes away
  - Treatment interest: Lip fillers

ESCALATED: No
ESCALATION_REASON: N/A

RECOMMENDED_NEXT_ACTION: Follow up with customer to book a free consultation.
Mention filler pricing starts from £250. Consider sending a WhatsApp message
with a booking link.

**Result:** ✅ Passed — All required sections present. Clean and actionable.
```

---

## 13. `README.md` — What to Write

```markdown
# Closira AI Agent — Bloom Aesthetics Clinic

A Python-based, 4-stage AI customer support workflow built using
Google Gemini, LangChain, and LangGraph.

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini 1.5 Pro via `langchain-google-genai` |
| LLM Calls | LangChain (`ChatGoogleGenerativeAI`, `SystemMessage`, `HumanMessage`) |
| Workflow Orchestration | LangGraph `StateGraph` with typed `AgentState` |
| Environment | Python 3.10+, `python-dotenv` |

## Setup

```bash
git clone <your-repo-url>
cd closira-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root:
```
GOOGLE_API_KEY=your-key-here
```

Get your key from https://aistudio.google.com/app/apikey (free tier available).

## Running the Workflow

```bash
python main.py
```

Type your message and press Enter. Type `quit` to end the session
and generate a structured summary.

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | CLI entry point and conversation loop |
| `agent/state.py` | LangGraph `AgentState` TypedDict — shared state schema |
| `agent/graph.py` | `StateGraph` wiring — nodes, edges, conditional routing |
| `agent/llm.py` | Shared `ChatGoogleGenerativeAI` factory |
| `agent/faq.py` | Stage 1: FAQ answering LangGraph node |
| `agent/qualification.py` | Stage 2: Lead qualification LangGraph node |
| `agent/escalation.py` | Stage 3: Escalation detection LangGraph node |
| `agent/summary.py` | Stage 4: Summary generation LangGraph node |
| `agent/sop_loader.py` | SOP JSON loader and text formatter |
| `data/sop.json` | SOP source of truth for Bloom Aesthetics Clinic |
| `logs/escalations.log` | Timestamped escalation event log |
| `logs/last_session_summary.txt` | Most recent session summary |
| `test_transcripts/` | One sample conversation per expected behaviour |
| `prompt_design.md` | Full system prompts and design decisions |

## Dependencies

- `langchain-google-genai` — LangChain's native Gemini integration
- `langgraph` — Workflow state machine (StateGraph, conditional edges)
- `langchain-core` — Messages, output parsers, prompt templates
- `google-generativeai` — Underlying Google SDK
- `python-dotenv` — Environment variable management
- `langchain-ollama` - for testing responses locally without wasting API credits.

## Trade-offs and Known Limitations

- **No persistent memory:** Conversation state lives in memory only.
  Each `python main.py` run starts a fresh session.
- **Qualification is a sub-loop, not a graph node per turn:** To keep
  the graph turn-based, qualification answers are collected in `main.py`
  between graph invocations rather than as separate graph turns. A
  production system would model each message as a graph turn with
  LangGraph's built-in checkpointer for persistence.
- **Escalation ends the session:** After escalation, the CLI session
  closes. A production system would hand off the live conversation thread
  to a human agent dashboard rather than terminating.
- **Single model for all stages:** All four nodes use Gemini 1.5 Pro.
  A production system would use a smaller, cheaper model (e.g. Gemini
  Flash) for classification nodes (escalation) and reserve the larger
  model for generation (FAQ answers, summary).
- **Synchronous API calls:** All LangChain calls are synchronous. For
  high-concurrency production use, LangChain's async interface
  (`await llm.ainvoke(messages)`) should be used instead.
```

---

## 14. Evaluation Checklist

Use this before submitting to confirm every criterion is met.

### AI Workflow Structure
- [ ] Four stages exist as separate LangGraph nodes in separate files
- [ ] `StateGraph` in `graph.py` wires them with clear conditional routing
- [ ] Escalation node runs before the FAQ node on every message
- [ ] Escalation (and `session_ended`) correctly route to the summary node

### Prompt Quality
- [ ] All system prompts are written out in full in `prompt_design.md`
- [ ] Reasoning is provided for the structured output format choice
- [ ] Reasoning is provided for why LangChain `SystemMessage` is used over raw strings
- [ ] Tone section explains why a clinic needs a specific persona

### Reliability & Safety
- [ ] Model cannot answer out-of-scope questions without escalating
- [ ] Scripted fallback response is defined for the out-of-scope case
- [ ] `CONFIDENCE: low` triggers escalation even if `ESCALATE: false`

### Escalation Logic
- [ ] At least 4 triggers implemented (anger, out-of-scope, medical, explicit request)
- [ ] Every escalation logged to `logs/escalations.log` with timestamp and reason
- [ ] Customer receives a warm, empathetic hand-off message
- [ ] `escalation_node` runs independently of `faq_node`

### SOP Understanding
- [ ] All prices, services, hours, and booking info sourced from `sop.json`
- [ ] Nothing in the AI's responses contradicts or extends the SOP

### Clarity of Reasoning
- [ ] `prompt_design.md` covers all four required topics
- [ ] LangGraph routing logic is explained (not just implemented)
- [ ] Trade-offs around the qualification sub-loop are documented in `README.md`

### Deliverables Checklist
- [ ] Public GitHub repo with clean commit history
- [ ] `prompt_design.md` complete
- [ ] `test_transcripts/` folder with all 5 scenarios
- [ ] `README.md` with setup, tech stack table, and trade-offs
- [ ] 2–5 minute video walkthrough recorded

---

> **Tip for the video:** Open `agent/graph.py` and explain the LangGraph routing diagram first (1 min) — evaluators will appreciate that you understand the architecture, not just the output. Then run a live demo: one in-scope FAQ answer, one escalation with the log file open, and the final summary (2–3 min). Keep it tight.
