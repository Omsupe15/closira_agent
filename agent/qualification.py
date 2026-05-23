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