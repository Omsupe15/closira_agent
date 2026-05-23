import logging
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from agent.llm import get_llm_with_fallback
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

If the customer asks a question about something that is in SOP, answer the question.

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
    llm = get_llm_with_fallback(temperature=0.2)   # Low temperature for deterministic classification

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