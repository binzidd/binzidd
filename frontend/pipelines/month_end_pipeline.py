"""
OpenWebUI Pipeline – Month-End Assistant.

OpenWebUI Pipelines allow you to intercept and transform messages before
they reach (or after they leave) any LLM.  This pipeline replaces the LLM
entirely, routing all requests through the Month-End Assistant backend.

Installation
────────────
  1. In OpenWebUI: Settings → Pipelines → Upload Pipeline
     Upload this file: month_end_pipeline.py
  2. Enable the pipeline for the "month-end-assistant" model
  3. Configure the Valves (BACKEND_URL etc.) in the Pipeline settings UI

How it works
────────────
  OpenWebUI calls pipe() with the user's messages.
  pipe() calls the FastAPI backend (/v1/chat/completions).
  The response streams back through OpenWebUI to the chat UI.

  ┌─────────────┐    messages     ┌────────────────┐    SSE stream
  │  OpenWebUI  │ ──────────────▶ │  This Pipeline │ ──────────────▶ backend
  │  Chat UI    │ ◀────────────── │  (pipe method) │ ◀────────────── FastAPI
  └─────────────┘                 └────────────────┘

OpenWebUI pipeline API reference:
  https://docs.openwebui.com/pipelines/
"""

from __future__ import annotations

import json
from typing import Any, Generator, Iterator, List, Optional, Union

import httpx
from pydantic import BaseModel, Field


class Pipeline:
    """
    Month-End Assistant OpenWebUI Pipeline.

    Valves (configurable via OpenWebUI UI or environment variables):
      BACKEND_URL       – FastAPI backend base URL
      DEFAULT_YEAR      – override accounting year (0 = use current year)
      DEFAULT_MONTH     – override accounting month (0 = use current month)
      DEFAULT_COMPANY   – default company ID
      STREAMING_ENABLED – stream tokens back to OpenWebUI
    """

    # ── OpenWebUI pipeline identity ───────────────────────────────────────────

    class Valves(BaseModel):
        """User-configurable settings exposed in the OpenWebUI pipeline UI."""

        BACKEND_URL:       str   = Field(default="http://localhost:8000", description="FastAPI backend URL")
        DEFAULT_YEAR:      int   = Field(default=0, description="Accounting year (0 = current)")
        DEFAULT_MONTH:     int   = Field(default=0, description="Accounting month (0 = current)")
        DEFAULT_COMPANY:   str   = Field(default="default-company", description="Company ID")
        STREAMING_ENABLED: bool  = Field(default=True, description="Stream tokens to OpenWebUI")
        REQUEST_TIMEOUT:   float = Field(default=120.0, description="HTTP timeout (seconds)")

    def __init__(self) -> None:
        self.name   = "Month-End Assistant"
        self.valves = self.Valves()
        self.id     = "month-end-assistant"
        self.type   = "pipe"

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    async def on_startup(self) -> None:
        """Called once when OpenWebUI loads the pipeline."""
        print(f"[{self.name}] Pipeline starting up. Backend: {self.valves.BACKEND_URL}")

    async def on_shutdown(self) -> None:
        """Called when OpenWebUI unloads the pipeline."""
        print(f"[{self.name}] Pipeline shutting down.")

    async def on_valves_updated(self) -> None:
        """Called whenever the user updates the Valves configuration."""
        print(f"[{self.name}] Valves updated. Backend: {self.valves.BACKEND_URL}")

    # ── Main pipeline method ──────────────────────────────────────────────────

    def pipe(
        self,
        user_message: str,
        model_id:     str,
        messages:     List[dict],
        body:         dict,
    ) -> Union[str, Generator[str, None, None], Iterator[str]]:
        """
        Main OpenWebUI pipeline entry point.

        Called for every user message.  Returns either:
          • A string (complete response)
          • A generator that yields string chunks (for streaming)

        Args:
            user_message: The latest user message content.
            model_id:     The selected model ID from the OpenWebUI dropdown.
            messages:     Full conversation history (role + content dicts).
            body:         Raw OpenAI-format request body.
        """
        import datetime as _dt

        now    = _dt.datetime.utcnow()
        year   = self.valves.DEFAULT_YEAR  or now.year
        month  = self.valves.DEFAULT_MONTH or now.month
        user   = body.get("user", {})
        uid    = user.get("email", "openwebui-user") if isinstance(user, dict) else "openwebui-user"

        # Parse period from user message if present (e.g. "march 2025")
        year, month = self._extract_period(user_message, year, month)

        # Determine backend model – strip "month-end-assistant." prefix if present
        backend_model = model_id.removeprefix("month-end-assistant.")

        payload = {
            "model":      backend_model,
            "messages":   messages,
            "stream":     self.valves.STREAMING_ENABLED,
            "user_id":    uid,
            "company_id": self.valves.DEFAULT_COMPANY,
            "year":       year,
            "month":      month,
        }

        if self.valves.STREAMING_ENABLED:
            return self._stream(payload)
        return self._fetch(payload)

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _stream(self, payload: dict) -> Generator[str, None, None]:
        """
        Stream SSE chunks from the FastAPI backend to OpenWebUI.

        Parses the 'data: {…}' lines and yields only the delta content
        so OpenWebUI's chat UI renders it progressively.
        """
        url = f"{self.valves.BACKEND_URL}/v1/chat/completions"
        with httpx.stream(
            "POST",
            url,
            json=payload,
            timeout=self.valves.REQUEST_TIMEOUT,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw = line.removeprefix("data: ").strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk   = json.loads(raw)
                    choices = chunk.get("choices", [{}])
                    delta   = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass

    def _fetch(self, payload: dict) -> str:
        """Non-streaming fetch – returns complete response as a single string."""
        url = f"{self.valves.BACKEND_URL}/v1/chat/completions"
        payload["stream"] = False
        try:
            response = httpx.post(
                url,
                json=payload,
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data    = response.json()
            choices = data.get("choices", [{}])
            return choices[0].get("message", {}).get("content", "No response from backend.")
        except httpx.HTTPError as exc:
            return f"⚠️ Backend error: {exc}"

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_period(message: str, default_year: int, default_month: int):
        """
        Attempt to extract a year/month from the user message.

        Handles patterns like: "march 2025", "2025-03", "03/2025"
        Falls back to defaults if no period is detected.
        """
        import re
        month_names = {
            "january": 1,  "february": 2,  "march": 3,     "april": 4,
            "may": 5,      "june": 6,      "july": 7,      "august": 8,
            "september": 9,"october": 10,  "november": 11, "december": 12,
        }
        msg = message.lower()

        # "march 2025" or "2025 march"
        for name, num in month_names.items():
            m = re.search(rf"{name}\s+(\d{{4}})|(\d{{4}})\s+{name}", msg)
            if m:
                year_str = m.group(1) or m.group(2)
                return int(year_str), num

        # "2025-03" or "03/2025"
        m = re.search(r"(\d{4})[-/](\d{1,2})|(\d{1,2})[-/](\d{4})", msg)
        if m:
            if m.group(1):
                return int(m.group(1)), int(m.group(2))
            return int(m.group(4)), int(m.group(3))

        return default_year, default_month


# ── Scenario-specific sub-pipelines ──────────────────────────────────────────

class AnomalyDetectionPipeline(Pipeline):
    """Pre-configured pipeline for the Anomaly Detection scenario."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Anomaly Detection"
        self.id   = "month-end-anomaly"

    def pipe(self, user_message, model_id, messages, body) -> Generator:
        import datetime as _dt
        now = _dt.datetime.utcnow()
        body.setdefault("year",  now.year)
        body.setdefault("month", now.month)
        body["model"] = "scenario-anomaly_detection"
        return super().pipe(user_message, "scenario-anomaly_detection", messages, body)


class RiskAssessmentPipeline(Pipeline):
    """Pre-configured pipeline for the Risk Assessment scenario."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Risk Assessment"
        self.id   = "month-end-risk"

    def pipe(self, user_message, model_id, messages, body) -> Generator:
        import datetime as _dt
        now = _dt.datetime.utcnow()
        body["model"] = "scenario-risk_assessment"
        return super().pipe(user_message, "scenario-risk_assessment", messages, body)
