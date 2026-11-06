"""Intent classification for chat, research reports and complete composition.

Routes user messages to GENERAL_CHAT, REPORT_GENERATION or WRITING.
"""
import json
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class IntentType(str, Enum):
    """User intent categories."""
    GENERAL_CHAT = "general_chat"
    REPORT_GENERATION = "report_generation"
    WRITING = "writing"


class IntentClassification(BaseModel):
    """Result of intent classification."""
    intent: IntentType
    confidence: Literal["high", "medium", "low"] = "high"
    reasoning: str = ""


# System prompts for automatic routing and ordinary chat
CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a research report application.

Your job: Select GENERAL_CHAT, REPORT_GENERATION, or WRITING.

WRITING indicators:
- Explicit 成文模式, complete essays, compositions, course assignments, reflections or speeches
- Requests to expand, polish or rewrite prose without an evidence-audited research requirement
- Prefer WRITING for ordinary articles/essays; prefer REPORT_GENERATION when research, source verification,
  auditable citations or a researched report is explicitly requested

REPORT_GENERATION indicators:
- Requests for reports, papers, documents, or analysis
- Research topics ("Write about X", "Analyze Y", "Research Z")
- Market analysis, technical documentation requests
- Phrases like: "create a report", "analyze", "research", "write a paper", "generate documentation"

GENERAL_CHAT indicators:
- Greetings ("hello", "hi", "hey")
- Questions about the app itself ("how does this work", "what can you do")
- General knowledge questions ("what is X", "tell me about Y")
- Conversational messages without a clear report/research intent
- Clarification questions ("can you help me", "I need assistance")

Respond ONLY with valid JSON in this exact format:
{
  "intent": "general_chat" or "report_generation" or "writing",
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation"
}

Examples:

User: "Write a report on quantum computing"
{"intent": "report_generation", "confidence": "high", "reasoning": "Explicit report request on specific topic"}

User: "Hello, how are you?"
{"intent": "general_chat", "confidence": "high", "reasoning": "Greeting with no research intent"}

User: "What can this app do?"
{"intent": "general_chat", "confidence": "high", "reasoning": "Question about app functionality"}

User: "Analyze the cryptocurrency market trends in 2024"
{"intent": "report_generation", "confidence": "high", "reasoning": "Analysis request implies detailed research report"}

User: "Tell me about machine learning"
{"intent": "general_chat", "confidence": "medium", "reasoning": "General knowledge question, could be brief answer or full report"}

Now classify the following user message:"""


CHAT_SYSTEM_PROMPT = """You are a helpful research assistant embedded in Report Studio, a desktop application for generating verified research reports.

Your role in chat mode:
- Answer questions clearly and concisely
- Help users understand how the app works
- Provide quick information without triggering full research pipelines
- Guide users on how to request reports when appropriate

Your capabilities:
- 成文模式 creates complete articles/essays, accepts supplied material as writing premises and checks internal contradictions; it does not certify factual truth.
- When users want a FULL RESEARCH REPORT with citations, web scraping, and PDF generation, tell them to phrase it as a clear report request (e.g., "Generate a report on X", "Analyze Y", "Research Z")
- For quick questions, provide helpful, accurate answers immediately
- You have access to the same LLM capabilities as the report generation, but you don't scrape the web or generate LaTeX

Be friendly, professional, and concise. If a user's question would be better served by the full report pipeline, suggest they rephrase it as a report request.

Keep responses under 300 words unless the user specifically asks for more detail."""


class LLMNotConfiguredError(RuntimeError):
    """Raised when no usable LLM provider is configured (missing API key, etc.).

    Distinct from a transient call failure: this needs a user-facing 'configure your
    API key in Settings' message, not a silent fallback to a mode that will also fail.
    """


async def classify_intent(message: str) -> IntentClassification:
    """Classify user message intent using the configured LLM.

    Raises LLMNotConfiguredError if the provider is not set up (so the endpoint can
    return a clear, actionable message). If the LLM is configured but the call or
    JSON parse fails, we fall back to GENERAL_CHAT rather than erroring the request.
    """
    from core_engine.report.llm import get_llm

    # get_llm() raises RuntimeError when the selected provider lacks its API key.
    # That is a configuration problem, surfaced distinctly — NOT a silent fallback.
    try:
        llm = get_llm()
    except RuntimeError as e:
        raise LLMNotConfiguredError(str(e)) from e

    # Explicit composition requests should not accidentally trigger costly report research.
    help_question = re.search(r"(?:什么是|如何使用|怎么使用|怎么切换|如何切换|有哪些|几个).{0,15}(?:模式|功能)", message)
    verified_request = re.search(r"逐句证据|事实核验|可核验.{0,6}(?:引文|引用)|研究报告|调研报告|"
                                 r"verified research|auditable citations|\b(?:research|report)\b", message, re.IGNORECASE)
    writing_request = re.search(r"成文模式|课程作业|心得体会|读后感|观后感|演讲稿|作文|扩写成文|"
                                r"写(?:一篇|篇).{0,35}(?:文章|随笔)|\b(?:essay|composition)\b", message, re.IGNORECASE)
    if writing_request and not help_question and ("成文模式" in message or not verified_request):
        return IntentClassification(intent=IntentType.WRITING, reasoning="明确要求完整文章或成文模式")

    try:
        # The LLM interface exposes _complete(system, user) for a raw completion.
        response = await llm.complete(
            system=CLASSIFIER_SYSTEM_PROMPT,
            user=message,
        )
        result = json.loads(response.strip())
        return IntentClassification(**result)
    except Exception as e:
        # LLM is configured but this call failed (bad JSON, transient error). Default
        # to chat so a report request isn't silently dropped — the chat handler will
        # surface any hard error to the user.
        return IntentClassification(
            intent=IntentType.GENERAL_CHAT,
            confidence="low",
            reasoning=f"Classification failed, defaulting to chat: {e}",
        )


async def handle_chat(message: str, history: str = "") -> str:
    """Handle general chat messages with a direct, context-aware LLM response.

    `history` is a rendered transcript of prior turns (see conversation.py). When
    present it is prepended so the assistant remembers the discussion; the latest
    user message is clearly delimited so the model knows what to answer.

    Raises LLMNotConfiguredError if the provider is not set up. Transient call errors
    return a friendly message string rather than raising.
    """
    from core_engine.report.llm import get_llm

    try:
        llm = get_llm()
    except RuntimeError as e:
        raise LLMNotConfiguredError(str(e)) from e

    if history:
        user_payload = (
            f"{history}\n\n"
            f"User: {message}\n\n"
            "Respond to the latest user message, using the conversation above for context."
        )
    else:
        user_payload = message

    try:
        response = await llm.complete(system=CHAT_SYSTEM_PROMPT, user=user_payload)
        # Guaranteed strip at the consumer boundary — defense-in-depth on top of the
        # provider-level strip, so no reasoning tags can ever reach the user.
        from core_engine.report.llm import strip_think_tags

        return strip_think_tags(response).strip()
    except Exception as e:
        return (
            f"I couldn't complete that request: {e}\n\n"
            "Please check your API key and Base URL in Settings, then try again."
        )

