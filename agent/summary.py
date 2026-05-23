from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from agent.llm import get_llm_with_fallback
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
    llm = get_llm_with_fallback(temperature=0.2)

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