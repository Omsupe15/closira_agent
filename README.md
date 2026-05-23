# Closira AI Agent — Bloom Aesthetics Clinic

A Python-based, 4-stage AI customer support workflow built using
Google Gemini, LangChain, and LangGraph.

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini flash-2.5 via `langchain-google-genai` |
| LLM Calls | LangChain (`ChatGoogleGenerativeAI`, `SystemMessage`, `HumanMessage`) |
| Workflow Orchestration | LangGraph `StateGraph` with typed `AgentState` |
| Environment | Python 3.10+, `python-dotenv` |
| Local Testing LLM | `ollama` (`gemma3:4b` is what i used.) |

## Setup

```bash
git clone https://github.com/Omsupe15/closira_agent.git
cd closira_agent
python -m venv .venv
source .venv/Scripts/activate      
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
| `tests/` | One sample conversation per expected behaviour |
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
  Each `python main.py` run starts a fresh session. In production we use a database to store the conversation history.
- **Qualification is a sub-loop, not a graph node per turn:** To keep
  the graph turn-based, qualification answers are collected in `main.py`
  between graph invocations rather than as separate graph turns. A
  production system would model each message as a graph turn with
  LangGraph's built-in checkpointer for persistence.
- **Escalation ends the session:** After escalation, the CLI session
  closes. A production system would hand off the live conversation thread
  to a human agent dashboard rather than terminating.
- **Single model for all stages:** All four nodes use Gemini flash-2.5.
  A production system would use a smaller, cheaper model for classification nodes (escalation) and reserve the larger
  model for generation (FAQ answers, summary).