from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from agent.llm import get_llm_with_fallback
from agent.state import AgentState
from agent.sop_loader import load_sop, sop_to_text

# Load SOP once at module level
_SOP = load_sop()
_SOP_TEXT = sop_to_text(_SOP)

FAQ_SYSTEM_PROMPT = f"""You are a helpful customer support assistant for {_SOP['business_name']}.

You must answer customer questions using ONLY the information in the SOP below.
Do not make up, infer, or estimate any information that is not explicitly stated in the SOP.
If the customer asks a question about something that is in SOP, answer the question.

Else If a customer asks something not covered in the SOP, respond with exactly:
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
    llm = get_llm_with_fallback(temperature=0.3)
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