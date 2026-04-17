"""
Streamlit UI for the Agentic Month-End Assistant.

Streams responses from the FastAPI backend via Server-Sent Events.
Set BACKEND_URL env var to point to your deployed backend (default: http://localhost:8000).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterator

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

MODELS: list[tuple[str, str]] = [
    ("month-end-assistant",              "Full Pipeline (LangGraph Orchestrator)"),
    ("month-end-supervisor",             "Multi-Agent Supervisor (5 workers)"),
    ("scenario-revenue_recognition",     "Scenario: Revenue Recognition"),
    ("scenario-anomaly_detection",       "Scenario: Anomaly Detection"),
    ("scenario-account_reconciliation",  "Scenario: Account Reconciliation"),
    ("scenario-accrual_calculation",     "Scenario: Accrual Calculation"),
    ("scenario-peer_benchmarking",       "Scenario: Peer Benchmarking"),
    ("scenario-cash_flow_forecast",      "Scenario: Cash-Flow Forecast"),
    ("scenario-audit_trail",             "Scenario: Audit Trail"),
    ("scenario-risk_assessment",         "Scenario: Risk Assessment"),
]

MONTH_NAMES = [datetime(2000, m, 1).strftime("%B") for m in range(1, 13)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def stream_sse(
    messages: list[dict],
    model: str,
    year: int,
    month: int,
    user_id: str,
    company_id: str,
) -> Iterator[str]:
    """Yield text chunks from the FastAPI SSE endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "year": year,
        "month": month,
        "user_id": user_id,
        "company_id": company_id,
    }
    with requests.post(
        f"{BACKEND_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


def submit_hitl_decision(
    thread_id: str,
    request_id: str,
    status: str,
    reviewer: str,
    comment: str,
) -> None:
    """POST a HITL approval decision to the backend."""
    if not thread_id or not request_id:
        st.sidebar.warning("Thread ID and Request ID are required.")
        return
    try:
        r = requests.post(
            f"{BACKEND_URL}/v1/month-end/approve",
            json={
                "thread_id": thread_id,
                "request_id": request_id,
                "status": status,
                "reviewer": reviewer,
                "comment": comment,
            },
            timeout=10,
        )
        if r.ok:
            st.sidebar.success(f"Decision submitted: **{status}**")
        else:
            st.sidebar.error(f"Backend error {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        st.sidebar.error(f"Request failed: {exc}")


def check_health() -> dict | None:
    """GET /health and return parsed JSON, or None on failure."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Month-End Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Month-End Assistant")
    st.divider()

    # Agent / model selector
    model_labels = [label for _, label in MODELS]
    model_ids    = [mid   for mid,  _ in MODELS]
    selected_label = st.selectbox("Agent / Scenario", model_labels, index=0)
    selected_model = model_ids[model_labels.index(selected_label)]

    # Accounting period
    st.subheader("Accounting Period")
    now   = datetime.utcnow()
    year  = st.number_input("Year",  value=now.year,  min_value=2020, max_value=now.year + 1, step=1)
    month = st.selectbox("Month", range(1, 13), index=now.month - 1,
                         format_func=lambda m: MONTH_NAMES[m - 1])

    st.divider()

    # Identity
    user_id    = st.text_input("User ID",    value="analyst-1")
    company_id = st.text_input("Company ID", value="acme-corp")

    st.divider()

    # Backend health
    if st.button("Check Backend Health", use_container_width=True):
        health = check_health()
        if health:
            st.success(f"Online — `{health.get('bedrock', 'unknown')}`")
            if health.get("agentcore"):
                st.info("AgentCore: connected")
            scenarios = health.get("scenarios", [])
            if scenarios:
                st.caption(f"{len(scenarios)} scenarios available")
        else:
            st.error(f"Cannot reach `{BACKEND_URL}`")

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # HITL approval panel
    st.divider()
    st.subheader("HITL Approval")
    hitl_thread  = st.text_input("Thread ID",  placeholder="from pipeline run output")
    hitl_request = st.text_input("Request ID", placeholder="from HITL notification")
    hitl_comment = st.text_input("Comment",    placeholder="Optional review note")

    col_approve, col_reject, col_escalate = st.columns(3)
    with col_approve:
        if st.button("Approve", type="primary", use_container_width=True):
            submit_hitl_decision(hitl_thread, hitl_request, "approved", user_id, hitl_comment)
    with col_reject:
        if st.button("Reject", use_container_width=True):
            submit_hitl_decision(hitl_thread, hitl_request, "rejected", user_id, hitl_comment)
    with col_escalate:
        if st.button("Escalate", use_container_width=True):
            submit_hitl_decision(hitl_thread, hitl_request, "escalated", user_id, hitl_comment)


# ── Main chat area ────────────────────────────────────────────────────────────

period_str = f"{MONTH_NAMES[int(month) - 1]} {int(year)}"
st.header("Month-End Close Assistant")
st.caption(
    f"Backend: `{BACKEND_URL}` | Agent: `{selected_model}` | Period: **{period_str}**"
)

# Initialise message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about your month-end close…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder    = st.empty()
        full_response  = ""
        try:
            for chunk in stream_sse(
                messages=[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages],
                model=selected_model,
                year=int(year),
                month=int(month),
                user_id=user_id,
                company_id=company_id,
            ):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
        except requests.exceptions.ConnectionError:
            full_response += f"\n\n⚠️ Cannot connect to backend at `{BACKEND_URL}`."
        except Exception as exc:
            full_response += f"\n\n⚠️ Error: {exc}"
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
