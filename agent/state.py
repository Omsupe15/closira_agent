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