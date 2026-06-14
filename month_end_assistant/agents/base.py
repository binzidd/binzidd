"""
Base Agent class.

Provides the shared Bedrock LLM instance, logging setup, and a thin wrapper
around langchain_aws.ChatBedrock so every sub-agent gets a consistent LLM
handle without repeating configuration.

All agents inherit from BaseAgent.  They do NOT call __init__ directly –
use the class-level `create()` factory so construction failures are handled
cleanly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from month_end_assistant.config import get_settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all Month-End Assistant agents.

    Attributes
    ──────────
        llm    – shared ChatBedrock instance
        tools  – list of LangChain tools bound to this agent
        name   – human-readable agent name used in logs and traces
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        name: str = "BaseAgent",
    ) -> None:
        self._settings = get_settings()
        self.name      = name
        self.tools     = tools or []
        self.llm       = self._build_llm()
        logger.debug("Agent '%s' initialised with %d tools.", name, len(self.tools))

    # ── LLM factory ──────────────────────────────────────────────────────────

    def _build_llm(self) -> BaseChatModel:
        """
        Instantiate the ChatBedrock LLM.

        Returns a stub object when AWS credentials are not configured so the
        rest of the graph can still be explored / unit-tested.
        """
        try:
            return ChatBedrock(
                model_id=self._settings.bedrock_model_id,
                region_name=self._settings.aws_region,
                model_kwargs={
                    "max_tokens": 4096,
                    "temperature": 0.1,   # low temperature for financial analysis
                },
                streaming=True,           # enables token streaming in the graph
            )
        except Exception as exc:
            logger.warning(
                "Could not connect to Bedrock (%s) – using stub LLM.", exc
            )
            return _StubLLM()

    # ── Helpers for sub-classes ───────────────────────────────────────────────

    def bind_tools(self, tools: List[BaseTool]) -> "BaseAgent":
        """Attach additional tools and return self (fluent interface)."""
        self.tools.extend(tools)
        return self

    def log_step(self, step: str, detail: str = "") -> None:
        """Structured step log shared by all agent nodes."""
        logger.info("[%s] %s %s", self.name, step, f"– {detail}" if detail else "")


# ─────────────────────────────────────────────────────────────────────────────
# Stub LLM (used when Bedrock is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class _StubLLM(BaseChatModel):
    """
    A deterministic stub that returns canned responses.

    Allows the graph structure and routing logic to be exercised without a
    live Bedrock connection – useful for unit tests and offline demos.
    """

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages: Any, stop: Any = None, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        last = messages[-1].content if messages else ""
        stub_text = (
            f"[StubLLM] Acknowledging: '{str(last)[:80]}…'\n"
            "This is a stub response – configure AWS credentials to use Bedrock."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=stub_text))])

    async def _agenerate(self, messages: Any, stop: Any = None, **kwargs: Any) -> Any:
        return self._generate(messages, stop, **kwargs)
