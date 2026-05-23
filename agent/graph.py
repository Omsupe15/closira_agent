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