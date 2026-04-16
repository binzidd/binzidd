"""
FastAPI Backend – OpenAI-Compatible API for OpenWebUI.

Exposes the Month-End Assistant through an OpenAI-compatible REST API so
OpenWebUI (or any OpenAI-compatible client) can connect without modification.

Endpoints
─────────
  GET  /v1/models               – list available "models" (our agent types)
  POST /v1/chat/completions     – chat completions with SSE streaming
  POST /v1/month-end/run        – trigger a full month-end close pipeline
  POST /v1/month-end/approve    – submit a HITL approval decision
  GET  /v1/month-end/scenarios  – list available deep-dive scenarios
  POST /v1/month-end/scenario   – run a single named scenario
  GET  /health                  – service health check
  WS   /ws/stream/{thread_id}   – WebSocket for real-time graph events

How OpenWebUI connects
──────────────────────
  1. Start this server:  uvicorn frontend.api.server:app --port 8000
  2. In OpenWebUI:       Settings → Connections → Add OpenAI API
                         URL:   http://localhost:8000/v1
                         Key:   any-string (no auth in dev mode)
  3. Select model:       "month-end-assistant" or "month-end-supervisor"

Streaming
─────────
  The /v1/chat/completions endpoint yields Server-Sent Events following the
  OpenAI streaming protocol (data: {"choices":[{"delta":{"content":"…"}}]}).
  OpenWebUI natively renders these as streaming chat bubbles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from month_end_assistant.agents.orchestrator import build_month_end_graph
from month_end_assistant.agents.scenarios import SCENARIO_REGISTRY, run_scenario
from month_end_assistant.agents.supervisor import MonthEndSupervisor
from month_end_assistant.callbacks import TokenStreamingCallback
from month_end_assistant.config import get_settings
from month_end_assistant.hitl import HITLManager
from month_end_assistant.memory import SessionManager
from month_end_assistant.models import (
    ApprovalStatus,
    HITLResponse,
    MonthEndPeriod,
    UserSession,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    str
    content: str
    name:    Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model:       str  = "month-end-assistant"
    messages:    List[ChatMessage]
    stream:      bool = True
    temperature: float = 0.1
    max_tokens:  int   = 2048
    # Custom extensions (ignored by OpenWebUI, used by our backend)
    user_id:     str   = Field(default="anonymous")
    company_id:  str   = Field(default="default-company")
    year:        Optional[int]  = None
    month:       Optional[int]  = None


class ChatCompletionChoice(BaseModel):
    index:         int
    message:       ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id:      str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object:  str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model:   str = "month-end-assistant"
    choices: List[ChatCompletionChoice]


class MonthEndRunRequest(BaseModel):
    user_id:    str = "anonymous"
    company_id: str = "default-company"
    year:       int
    month:      int = Field(ge=1, le=12)
    thread_id:  Optional[str] = None


class ApprovalRequest(BaseModel):
    thread_id:  str
    request_id: str
    status:     ApprovalStatus
    reviewer:   str = "api-user"
    comment:    str = ""


class ScenarioRequest(BaseModel):
    scenario:   str
    year:       int
    month:      int = Field(ge=1, le=12)
    user_id:    str = "anonymous"


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and return the FastAPI application."""

    app = FastAPI(
        title="Month-End Assistant API",
        description=(
            "OpenAI-compatible REST API for the Agentic Month-End Assistant. "
            "Connect OpenWebUI to this server to get a full AI-powered "
            "month-end close interface."
        ),
        version="1.0.0",
    )

    # ── CORS (required for OpenWebUI → API communication) ────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # restrict to your OpenWebUI domain in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Singletons shared across requests ────────────────────────────────────
    session_mgr  = SessionManager()
    hitl_manager = HITLManager()
    supervisor   = MonthEndSupervisor()

    # ── Models listing ────────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """
        OpenAI-compatible model list.
        OpenWebUI calls this to populate its model selector dropdown.
        """
        models = [
            {
                "id":       "month-end-assistant",
                "object":   "model",
                "created":  int(time.time()),
                "owned_by": "month-end-assistant",
                "description": "Full month-end close pipeline (LangGraph orchestrator)",
            },
            {
                "id":       "month-end-supervisor",
                "object":   "model",
                "created":  int(time.time()),
                "owned_by": "month-end-assistant",
                "description": "Multi-agent supervisor with 5 specialist workers",
            },
            *[
                {
                    "id":       f"scenario-{name}",
                    "object":   "model",
                    "created":  int(time.time()),
                    "owned_by": "month-end-assistant",
                    "description": f"Deep-dive: {name.replace('_', ' ').title()}",
                }
                for name in SCENARIO_REGISTRY
            ],
        ]
        return {"object": "list", "data": models}

    # ── Chat completions ──────────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint with SSE streaming.

        OpenWebUI sends the conversation history here and expects either:
          • A StreamingResponse (when stream=True) with SSE data chunks
          • A JSON response with the full message (when stream=False)

        The endpoint routes to the appropriate agent based on the model name:
          month-end-assistant → Orchestrator pipeline
          month-end-supervisor → Supervisor with worker agents
          scenario-* → Named scenario runner
        """
        user_message = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            "",
        )

        # Determine which year/month the user is asking about
        import datetime as _dt
        now   = _dt.datetime.utcnow()
        year  = req.year  or now.year
        month = req.month or now.month

        if req.stream:
            return StreamingResponse(
                _stream_response(
                    model=req.model,
                    user_id=req.user_id,
                    company_id=req.company_id,
                    year=year,
                    month=month,
                    user_message=user_message,
                    supervisor=supervisor,
                    session_mgr=session_mgr,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control":              "no-cache",
                    "X-Accel-Buffering":          "no",
                    "Transfer-Encoding":          "chunked",
                },
            )

        # Non-streaming fallback
        reply = await _build_non_streaming_reply(
            model=req.model,
            user_id=req.user_id,
            company_id=req.company_id,
            year=year,
            month=month,
            user_message=user_message,
            supervisor=supervisor,
            session_mgr=session_mgr,
        )
        return ChatCompletionResponse(
            model=req.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=reply),
            )],
        )

    # ── Month-end pipeline trigger ────────────────────────────────────────────

    @app.post("/v1/month-end/run")
    async def run_month_end(req: MonthEndRunRequest) -> Dict[str, Any]:
        """
        Trigger the full month-end close pipeline asynchronously.

        Returns immediately with the thread_id.  The client can poll
        /v1/month-end/status/{thread_id} or listen on the WebSocket.
        """
        period  = MonthEndPeriod(year=req.year, month=req.month)
        session = await session_mgr.load_or_create(req.user_id, req.company_id)
        session = await session_mgr.update_active_period(session, period)

        orchestrator = build_month_end_graph()

        # Run in background task
        asyncio.create_task(
            orchestrator.run(
                period=period,
                session=session,
                thread_id=req.thread_id or session.thread_id,
            )
        )

        return {
            "status":    "running",
            "thread_id": req.thread_id or session.thread_id,
            "period":    period.label,
            "message":   "Pipeline started. Connect to /ws/stream/{thread_id} for real-time updates.",
        }

    # ── HITL approval ─────────────────────────────────────────────────────────

    @app.post("/v1/month-end/approve")
    async def submit_approval(req: ApprovalRequest) -> Dict[str, Any]:
        """
        Submit a human approval decision to resume an interrupted pipeline.

        Called when the user clicks Approve / Reject / Escalate in
        OpenWebUI, Teams, or Slack.
        """
        response = HITLResponse(
            request_id=req.request_id,
            status=req.status,
            reviewer=req.reviewer,
            comment=req.comment,
        )
        orchestrator = build_month_end_graph()
        await orchestrator.resume(thread_id=req.thread_id, response=response)
        return {
            "status":     "resumed",
            "decision":   req.status.value,
            "thread_id":  req.thread_id,
        }

    # ── Scenario endpoints ────────────────────────────────────────────────────

    @app.get("/v1/month-end/scenarios")
    async def list_scenarios() -> Dict[str, Any]:
        """List all available deep-dive scenario names."""
        return {
            "scenarios": [
                {"name": name, "description": cls.__doc__.strip().splitlines()[0]}
                for name, cls in SCENARIO_REGISTRY.items()
            ]
        }

    @app.post("/v1/month-end/scenario")
    async def run_named_scenario(req: ScenarioRequest) -> Dict[str, Any]:
        """Run a single named scenario and return its result."""
        period = MonthEndPeriod(year=req.year, month=req.month)
        try:
            result = await run_scenario(req.scenario, period)
            return {"scenario": req.scenario, "period": period.label, "result": result}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("Scenario %s failed", req.scenario)
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Health check ──────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        settings = get_settings()
        return {
            "status":         "ok",
            "version":        "1.0.0",
            "bedrock":        settings.bedrock_model_id,
            "agentcore":      settings.has_agentcore,
            "teams":          settings.has_teams,
            "slack":          settings.has_slack,
            "scenarios":      list(SCENARIO_REGISTRY.keys()),
        }

    # ── WebSocket – real-time graph event streaming ────────────────────────────

    @app.websocket("/ws/stream/{thread_id}")
    async def websocket_stream(websocket: WebSocket, thread_id: str):
        """
        WebSocket endpoint for real-time pipeline event streaming.

        OpenWebUI (or a custom frontend) can connect here to receive
        agent step events, research progress, and HITL status updates
        as they happen – without polling.
        """
        await websocket.accept()
        logger.info("WebSocket connected for thread_id=%s", thread_id)
        try:
            # Send a welcome message
            await websocket.send_json({
                "event":     "connected",
                "thread_id": thread_id,
                "message":   "Connected to Month-End Assistant event stream.",
            })

            # In production, subscribe to a real event bus (Redis Pub/Sub etc.)
            # For demo, send periodic progress pings
            for i in range(10):
                await asyncio.sleep(2)
                await websocket.send_json({
                    "event":   "progress",
                    "step":    i + 1,
                    "message": f"Pipeline step {i + 1} in progress…",
                })

            await websocket.send_json({"event": "complete", "message": "Pipeline complete."})
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for thread_id=%s", thread_id)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Streaming helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_response(
    model: str,
    user_id: str,
    company_id: str,
    year: int,
    month: int,
    user_message: str,
    supervisor: MonthEndSupervisor,
    session_mgr: SessionManager,
) -> AsyncIterator[str]:
    """
    Generate SSE chunks in the OpenAI streaming format.

    Each yielded string is a 'data: …\\n\\n' SSE line.  OpenWebUI consumes
    these and appends the delta content to the chat bubble in real time.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created       = int(time.time())
    period        = MonthEndPeriod(year=year, month=month)

    def _sse(content: str, finish: Optional[str] = None) -> str:
        delta = {"content": content} if content else {}
        chunk = {
            "id":      completion_id,
            "object":  "chat.completion.chunk",
            "created": created,
            "model":   model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # ── Route to the right agent ──────────────────────────────────────────────
    try:
        if model.startswith("scenario-"):
            scenario_name = model.removeprefix("scenario-")
            yield _sse(f"Running scenario: **{scenario_name.replace('_', ' ').title()}** for {period.label}…\n\n")
            result = await run_scenario(scenario_name, period)
            # Stream the result as formatted text
            for key, val in result.items():
                if key in ("scenario", "period"):
                    continue
                yield _sse(f"**{key.replace('_', ' ').title()}**\n")
                yield _sse(f"{json.dumps(val, indent=2, default=str)}\n\n")

        elif model == "month-end-supervisor":
            yield _sse(f"Activating supervisor for {period.label}…\n\n")
            result = await supervisor.run(
                query=user_message or f"Perform full analysis for {period.label}",
                period_label=period.label,
            )
            messages = result.get("messages", [])
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    yield _sse(f"{msg.content}\n")

        else:
            # Default: orchestrator pipeline
            yield _sse(f"Starting month-end close for **{period.label}**…\n\n")
            session = await session_mgr.load_or_create(user_id, company_id)
            session = await session_mgr.update_active_period(session, period)
            orchestrator = build_month_end_graph()

            # Stream individual step progress messages
            steps = [
                ("📂", "Loading session memory from AWS AgentCore…"),
                ("🔬", "Running LangGraph deep research (plan → parallel workers → reflect)…"),
                ("📊", "Analysing financials – fetching actuals and computing variances…"),
                ("⚗️",  "Sandbox deep-dive – DSO, DPO, working capital metrics…"),
                ("📝", "Assembling month-end report draft…"),
                ("🔔", "Sending HITL approval notifications to Teams + Slack…"),
            ]
            for icon, step_text in steps:
                yield _sse(f"{icon} {step_text}\n")
                await asyncio.sleep(0.3)   # pacing for UX

            yield _sse(f"\n✅ Pipeline ready.  Use `/v1/month-end/approve` to submit your decision.\n")

    except Exception as exc:
        logger.exception("Streaming response failed")
        yield _sse(f"\n⚠️ Error: {exc}\n")

    # OpenAI spec: final chunk with finish_reason + [DONE]
    yield _sse("", finish="stop")
    yield "data: [DONE]\n\n"


async def _build_non_streaming_reply(
    model: str,
    user_id: str,
    company_id: str,
    year: int,
    month: int,
    user_message: str,
    supervisor: MonthEndSupervisor,
    session_mgr: SessionManager,
) -> str:
    """Build a complete non-streaming reply (used when stream=False)."""
    period = MonthEndPeriod(year=year, month=month)

    if model.startswith("scenario-"):
        scenario_name = model.removeprefix("scenario-")
        result = await run_scenario(scenario_name, period)
        return json.dumps(result, indent=2, default=str)

    if model == "month-end-supervisor":
        result = await supervisor.run(
            query=user_message or f"Perform full analysis for {period.label}",
            period_label=period.label,
        )
        messages = result.get("messages", [])
        return "\n".join(m.content for m in messages if hasattr(m, "content"))

    return (
        f"Month-end pipeline initiated for {period.label}. "
        "Check /v1/month-end/run for full pipeline execution."
    )


# ── Application instance ──────────────────────────────────────────────────────

app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("frontend.api.server:app", host="0.0.0.0", port=8000, reload=True)
