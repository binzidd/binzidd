"""
Streamlit UI for the Agentic Month-End Assistant.

Authentication flow
───────────────────
  1. Show login form (username + password + arithmetic CAPTCHA).
  2. Validate password against bcrypt hash stored in APP_PASSWORD_HASH env var.
  3. On success set st.session_state.authenticated = True and show the main app.

Environment variables
─────────────────────
  APP_USERNAME        Login username  (default: mea_admin)
  APP_PASSWORD_HASH   bcrypt hash of the password
  BACKEND_URL         FastAPI backend base URL (default: http://localhost:8000)

Generate a new password hash:
  python3 -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt(12)).decode())"
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Iterator

import bcrypt
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND_URL   = os.getenv("BACKEND_URL",   "http://localhost:8000").rstrip("/")
APP_USERNAME  = os.getenv("APP_USERNAME",  "mea_admin")
APP_PWD_HASH  = os.getenv("APP_PASSWORD_HASH", "")

MODELS: list[tuple[str, str]] = [
    ("month-end-assistant",             "Full Pipeline (LangGraph Orchestrator)"),
    ("month-end-supervisor",            "Multi-Agent Supervisor (5 workers)"),
    ("scenario-revenue_recognition",    "Scenario: Revenue Recognition"),
    ("scenario-anomaly_detection",      "Scenario: Anomaly Detection"),
    ("scenario-account_reconciliation", "Scenario: Account Reconciliation"),
    ("scenario-accrual_calculation",    "Scenario: Accrual Calculation"),
    ("scenario-peer_benchmarking",      "Scenario: Peer Benchmarking"),
    ("scenario-cash_flow_forecast",     "Scenario: Cash-Flow Forecast"),
    ("scenario-audit_trail",            "Scenario: Audit Trail"),
    ("scenario-risk_assessment",        "Scenario: Risk Assessment"),
]

MONTH_NAMES = [datetime(2000, m, 1).strftime("%B") for m in range(1, 13)]

MAX_ATTEMPTS  = 5    # lock out after this many consecutive failures
LOCKOUT_SECS  = 300  # 5-minute lockout window


# ── Page configuration (must come first) ─────────────────────────────────────

st.set_page_config(
    page_title="Month-End Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def _init_auth_state() -> None:
    """Initialise session-state keys used by the auth layer."""
    defaults = {
        "authenticated":  False,
        "login_attempts": 0,
        "lockout_until":  0.0,
        "captcha_a":      random.randint(1, 12),
        "captcha_b":      random.randint(1, 12),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _rotate_captcha() -> None:
    """Generate a new CAPTCHA challenge."""
    st.session_state.captcha_a = random.randint(1, 12)
    st.session_state.captcha_b = random.randint(1, 12)


# ── Login page ────────────────────────────────────────────────────────────────

def show_login() -> None:
    """Render the login form. Sets session_state.authenticated on success."""
    _init_auth_state()

    # Centre the login card
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("## 📊 Month-End Assistant")
        st.markdown("##### Secure Login")
        st.divider()

        # Lockout check
        remaining = st.session_state.lockout_until - time.time()
        if remaining > 0:
            st.error(
                f"Too many failed attempts. Try again in **{int(remaining // 60) + 1} minute(s)**."
            )
            st.stop()

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            # Arithmetic CAPTCHA
            st.markdown("**Security check**")
            a = st.session_state.captcha_a
            b = st.session_state.captcha_b
            captcha_answer = st.number_input(
                f"What is {a} + {b}?",
                min_value=0,
                max_value=99,
                step=1,
                value=0,
            )

            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

        if submitted:
            # Validate CAPTCHA first (no bcrypt call on bot traffic)
            if int(captcha_answer) != (a + b):
                st.error("Incorrect CAPTCHA answer. Please try again.")
                _rotate_captcha()
                st.rerun()
                return

            # Validate credentials
            if username == APP_USERNAME and _verify_password(password, APP_PWD_HASH):
                st.session_state.authenticated  = True
                st.session_state.login_attempts = 0
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                _rotate_captcha()
                remaining_tries = MAX_ATTEMPTS - st.session_state.login_attempts
                if st.session_state.login_attempts >= MAX_ATTEMPTS:
                    st.session_state.lockout_until = time.time() + LOCKOUT_SECS
                    st.error("Account locked for 5 minutes due to repeated failures.")
                else:
                    st.error(
                        f"Invalid username or password. "
                        f"{remaining_tries} attempt(s) remaining."
                    )
                st.rerun()


# ── Main-app helpers ──────────────────────────────────────────────────────────

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
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "year":       year,
        "month":      month,
        "user_id":    user_id,
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
                "thread_id":  thread_id,
                "request_id": request_id,
                "status":     status,
                "reviewer":   reviewer,
                "comment":    comment,
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


# ── Main application ──────────────────────────────────────────────────────────

def show_app() -> None:
    """Render the full Month-End Assistant UI (authenticated users only)."""

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📊 Month-End Assistant")
        st.caption(f"Signed in as **{APP_USERNAME}**")
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
        month = st.selectbox(
            "Month", range(1, 13), index=now.month - 1,
            format_func=lambda m: MONTH_NAMES[m - 1],
        )

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

        # Logout
        if st.button("Log Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
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

    # ── Main chat area ────────────────────────────────────────────────────────
    period_str = f"{MONTH_NAMES[int(month) - 1]} {int(year)}"
    st.header("Month-End Close Assistant")
    st.caption(
        f"Backend: `{BACKEND_URL}` | Agent: `{selected_model}` | Period: **{period_str}**"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your month-end close…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""
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


# ── Entry point ───────────────────────────────────────────────────────────────

_init_auth_state()

if st.session_state.authenticated:
    show_app()
else:
    show_login()
