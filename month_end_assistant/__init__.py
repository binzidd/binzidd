"""
Agentic Month-End Assistant
===========================

A production-grade AI assistant for automating the financial month-end close
process, built on:

  • LangGraph   – deep research agent with reflection loops, parallel execution
                  via the Send API, and interrupt()-based HITL checkpointing
  • AWS AgentCore – managed agent runtime, persistent Memory, and tool execution
  • MS Teams    – Adaptive Card approval notifications
  • Slack       – Block Kit approval notifications
  • RestrictedPython sandbox – safe execution of custom financial calculations
  • Pydantic v2 / pydantic-settings – fully typed, env-driven configuration

Quick start
───────────
  from month_end_assistant import MonthEndPeriod
  from month_end_assistant.agents import build_month_end_graph
  from month_end_assistant.memory import SessionManager

  # See main.py for the full demo.
"""

__version__ = "1.0.0"
__author__  = "Agentic Month-End Assistant"

from month_end_assistant.models import MonthEndPeriod  # noqa: F401 – convenience re-export
