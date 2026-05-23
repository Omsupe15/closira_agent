import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Primary LLM — Google Gemini 1.5 Pro via LangChain.
    Use temperature=0.1 for classification nodes (escalation),
    temperature=0.3 for generation nodes (FAQ, summary).
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def get_fallback_llm(temperature: float = 0.3) -> ChatOllama:
    """
    Fallback LLM — Gemma 3:4b running locally via Ollama.
    Requires Ollama to be installed and running on localhost:11434.
    Pull the model first with: ollama pull gemma3:4b
    """
    return ChatOllama(
        model="gemma3:4b",
        temperature=temperature,
        base_url="http://localhost:11434"
    )


def get_llm_with_fallback(temperature: float = 0.3) -> BaseChatModel:
    """
    Returns Gemini if the API key is present and reachable.
    Automatically falls back to local Gemma 3:4b via Ollama if:
      - GOOGLE_API_KEY is missing or empty
      - Gemini raises any exception on a test ping

    Usage: replace get_llm() with get_llm_with_fallback() in any node
    where you want automatic resilience.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        logger.warning(
            "GOOGLE_API_KEY not set — falling back to local Gemma 3:4b via Ollama."
        )
        return get_fallback_llm(temperature)

    # Attempt a minimal ping to verify the Gemini API is reachable
    try:
        gemini = get_llm(temperature)
        gemini.invoke("ping")   # Single-token test call
        return gemini

    except Exception as e:
        logger.warning(
            f"Gemini unavailable ({type(e).__name__}: {e}) — "
            f"falling back to local Gemma 3:4b via Ollama."
        )
        return get_fallback_llm(temperature)