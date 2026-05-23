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
        if state.get("faq_result", {}).get("answer") and state["faq_result"]["answer"] != state.get("next_agent_message"):
            print(f"\nAgent: {state['faq_result']['answer']}")
        if state.get("next_agent_message"):
            print(f"\nAgent: {state['next_agent_message']}")


        # ── If escalated, end session ─────────────────────────────────────
        if state.get("escalated"):
            break
        if not state.get("escalated"):
            faq_msg=state.get("next_agent_message")
            

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
    

    # Save to file
    os.makedirs("logs", exist_ok=True)
    with open("logs/last_session_summary.txt", "w") as f:
        f.write(state["summary"])
    print("\nYour request has been noted our team will contact you shortly.")    


if __name__ == "__main__":
    run_conversation()