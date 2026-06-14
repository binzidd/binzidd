"""
Agent and tool registry with capability discovery and load-based selection.

The registry is the harness's service-locator: every agent and tool registers
itself at startup.  When a task arrives the harness queries `discover()` to
find capable agents, then `select_best()` to choose the one with the highest
health score (success rate × speed).

Usage::
    registry = AgentRegistry()
    registry.register_agent(
        orchestrator,
        AgentCapability(
            name="orchestrator",
            description="Full month-end close pipeline with HITL",
            tags={"monthly", "quarterly", "full-pipeline"},
            input_keys=["task", "period", "user_id"],
            output_keys=["report"],
        )
    )
    # Later:
    caps = registry.discover("revenue analysis")
    agent, cap = registry.select_best("revenue")
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    name: str
    description: str
    tags: Set[str] = field(default_factory=set)
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    total_runs: int = 0

    def health_score(self) -> float:
        """Combined score: high success rate + low latency."""
        latency_penalty = min(1.0, self.avg_latency_ms / 60_000)  # normalise to 60s
        return self.success_rate * (1.0 - 0.3 * latency_penalty)

    def matches(self, query: str) -> bool:
        q = query.lower()
        return (
            any(tag in q for tag in self.tags)
            or any(word in self.description.lower() for word in q.split())
        )


@dataclass
class ToolCapability:
    name: str
    description: str
    tags: Set[str] = field(default_factory=set)
    call_count: int = 0
    error_count: int = 0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.call_count if self.call_count else 0.0


class AgentRegistry:
    """
    Central registry for agents and tools.

    Thread-safe for reads; write operations (register/record) use a simple
    dict which is fine for single-process deployments.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Tuple[Any, AgentCapability]] = {}
        self._tools: Dict[str, Tuple[Any, ToolCapability]] = {}
        self._run_history: Dict[str, List[Dict]] = defaultdict(list)

    # ── Agent registration ────────────────────────────────────────────────────

    def register_agent(self, agent: Any, capability: AgentCapability) -> None:
        self._agents[capability.name] = (agent, capability)
        logger.info("Registry: agent '%s' registered (tags=%s)", capability.name, capability.tags)

    def unregister_agent(self, name: str) -> None:
        self._agents.pop(name, None)

    def get_agent(self, name: str) -> Optional[Any]:
        pair = self._agents.get(name)
        return pair[0] if pair else None

    def get_capability(self, name: str) -> Optional[AgentCapability]:
        pair = self._agents.get(name)
        return pair[1] if pair else None

    # ── Tool registration ─────────────────────────────────────────────────────

    def register_tool(self, tool: Any, capability: ToolCapability) -> None:
        self._tools[capability.name] = (tool, capability)

    def get_tool(self, name: str) -> Optional[Any]:
        pair = self._tools.get(name)
        return pair[0] if pair else None

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover(self, query: str) -> List[AgentCapability]:
        """Return all agents whose capabilities match the query string."""
        return [
            cap for _, (_, cap) in self._agents.items()
            if cap.matches(query)
        ]

    def discover_tools(self, query: str) -> List[ToolCapability]:
        return [
            cap for _, (_, cap) in self._tools.items()
            if any(tag in query.lower() for tag in cap.tags)
               or query.lower() in cap.description.lower()
        ]

    def select_best(self, query: str) -> Optional[Tuple[Any, AgentCapability]]:
        """
        Find the best-suited agent for a query using health score ranking.

        Falls back to the 'orchestrator' if no match found.
        """
        candidates = [
            (agent, cap)
            for _, (agent, cap) in self._agents.items()
            if cap.matches(query)
        ]
        if not candidates:
            default = self._agents.get("orchestrator")
            if default:
                logger.debug("No match for '%s' — using orchestrator", query)
                return default
            return None

        return max(candidates, key=lambda pair: pair[1].health_score())

    # ── Metrics recording ─────────────────────────────────────────────────────

    def record_run(
        self,
        agent_name: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Update agent metrics after a run completes."""
        cap = self.get_capability(agent_name)
        if not cap:
            return

        cap.total_runs += 1
        # Exponential moving average for latency (α=0.2)
        if cap.avg_latency_ms == 0:
            cap.avg_latency_ms = latency_ms
        else:
            cap.avg_latency_ms = 0.8 * cap.avg_latency_ms + 0.2 * latency_ms

        # EMA for success rate
        new_rate = 1.0 if success else 0.0
        cap.success_rate = 0.9 * cap.success_rate + 0.1 * new_rate

        self._run_history[agent_name].append({
            "ts": time.monotonic(),
            "latency_ms": latency_ms,
            "success": success,
        })
        # Keep only last 100 runs in memory
        if len(self._run_history[agent_name]) > 100:
            self._run_history[agent_name].pop(0)

    def record_tool_call(self, tool_name: str, success: bool) -> None:
        pair = self._tools.get(tool_name)
        if not pair:
            return
        _, cap = pair
        cap.call_count += 1
        if not success:
            cap.error_count += 1

    # ── Status report ─────────────────────────────────────────────────────────

    def status_report(self) -> str:
        lines = ["── Agent Registry Status ──────────────────────"]
        for name, (_, cap) in self._agents.items():
            lines.append(
                f"  {name:<25} health={cap.health_score():.2f}  "
                f"runs={cap.total_runs}  p50={cap.avg_latency_ms:.0f}ms  "
                f"success={cap.success_rate:.0%}"
            )
        if self._tools:
            lines.append("")
            lines.append("── Tool Registry ──────────────────────────────")
            for name, (_, cap) in self._tools.items():
                lines.append(
                    f"  {name:<25} calls={cap.call_count}  "
                    f"errors={cap.error_count}  error_rate={cap.error_rate:.0%}"
                )
        return "\n".join(lines)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
