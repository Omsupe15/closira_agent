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