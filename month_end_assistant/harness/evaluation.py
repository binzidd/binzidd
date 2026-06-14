"""
Agent evaluation framework.

Provides LLM-as-judge evaluation with structured scoring, dataset management,
and regression reporting.  Designed to be run in CI or interactively.

Evaluators available:
  correctness   – does the output answer the question correctly?
  completeness  – are all required elements present?
  groundedness  – are claims supported by the provided context?
  conciseness   – is the answer appropriately concise?
  hallucination – does the output invent facts not in the context?
  relevance     – is the output relevant to the input?

Usage::
    evaluator = AgentEvaluator(llm=llm, settings=settings)
    evaluator.add_case("default",
        input={"task": "Analyse Q3 revenue variance"},
        expected={"contains": ["variance", "IFRS 15"]},
        metadata={"scenario": "revenue_recognition"},
    )
    report = await evaluator.evaluate(run_fn=agent.run, dataset="default")
    print(report.summary())
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from month_end_assistant.harness.config import HarnessSettings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    input: Dict[str, Any]
    expected: Dict[str, Any]                  # can contain "contains", "schema", "score_min"
    metadata: Dict[str, Any] = field(default_factory=dict)
    case_id: str = field(default_factory=lambda: f"case-{int(time.monotonic()*1000)}")


@dataclass
class EvalScore:
    evaluator: str
    score: float                               # 0.0–1.0
    reasoning: str = ""
    passed: bool = False

    def __post_init__(self):
        self.passed = self.score >= 0.7


@dataclass
class EvalResult:
    case: EvalCase
    output: Any
    scores: List[EvalScore]
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def avg_score(self) -> float:
        return sum(s.score for s in self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores)


@dataclass
class EvalReport:
    dataset: str
    results: List[EvalResult]
    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pass_threshold: float = 0.75

    @property
    def avg_score(self) -> float:
        return sum(r.avg_score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0.0

    @property
    def passed(self) -> bool:
        return self.avg_score >= self.pass_threshold

    def summary(self) -> str:
        lines = [
            f"── Eval Report: {self.dataset} ──────────────────────",
            f"  Cases:       {len(self.results)}",
            f"  Avg Score:   {self.avg_score:.2f}",
            f"  Pass Rate:   {self.pass_rate:.0%}",
            f"  Status:      {'✅ PASS' if self.passed else '❌ FAIL'}",
            f"  Run at:      {self.run_at}",
            "",
        ]
        for i, result in enumerate(self.results, 1):
            status = "✅" if result.passed else "❌"
            lines.append(f"  [{status}] Case {i}: avg={result.avg_score:.2f}  latency={result.latency_ms:.0f}ms")
            for score in result.scores:
                lines.append(f"       {score.evaluator}: {score.score:.2f} – {score.reasoning[:80]}")
            if result.error:
                lines.append(f"       ERROR: {result.error}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-judge prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an expert evaluator of AI agent outputs for financial analysis tasks. "
    "Score responses on a scale of 0.0 to 1.0. "
    "Return ONLY a JSON object: {\"score\": <float>, \"reasoning\": \"<one sentence>\"}."
)

_EVAL_PROMPTS: Dict[str, str] = {
    "correctness": (
        "Evaluate whether the OUTPUT correctly answers the question in INPUT.\n"
        "If EXPECTED contains specific facts, check they appear in OUTPUT.\n\n"
        "INPUT: {input}\nOUTPUT: {output}\nEXPECTED: {expected}"
    ),
    "completeness": (
        "Evaluate whether the OUTPUT covers all elements expected for INPUT.\n"
        "Missing elements should lower the score significantly.\n\n"
        "INPUT: {input}\nOUTPUT: {output}\nEXPECTED: {expected}"
    ),
    "groundedness": (
        "Evaluate whether every claim in OUTPUT is grounded in the INPUT context.\n"
        "Ungrounded or speculative claims should lower the score.\n\n"
        "INPUT: {input}\nOUTPUT: {output}"
    ),
    "conciseness": (
        "Evaluate whether OUTPUT is appropriately concise for a financial analysis task.\n"
        "Excessive repetition or padding should lower the score.\n\n"
        "INPUT: {input}\nOUTPUT: {output}"
    ),
    "hallucination": (
        "Evaluate whether OUTPUT contains fabricated facts not derivable from INPUT.\n"
        "0.0 = heavily hallucinated, 1.0 = fully grounded.\n\n"
        "INPUT: {input}\nOUTPUT: {output}"
    ),
    "relevance": (
        "Evaluate whether OUTPUT is relevant to the financial analysis requested in INPUT.\n\n"
        "INPUT: {input}\nOUTPUT: {output}"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class AgentEvaluator:
    """LLM-as-judge evaluator with dataset management."""

    def __init__(self, llm: BaseChatModel, settings: HarnessSettings) -> None:
        self._llm = llm
        self._settings = settings
        self._datasets: Dict[str, List[EvalCase]] = {}
        self._history: List[EvalReport] = []

    # ── Dataset management ────────────────────────────────────────────────────

    def add_case(
        self,
        dataset: str,
        input: Dict[str, Any],
        expected: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvalCase:
        case = EvalCase(input=input, expected=expected, metadata=metadata or {})
        self._datasets.setdefault(dataset, []).append(case)
        return case

    def add_default_financial_cases(self) -> None:
        """Seed the default dataset with common month-end analysis test cases."""
        ds = self._settings.eval_default_dataset

        self.add_case(ds,
            input={"task": "Analyse revenue variance for Q3 2025", "period": "2025-09"},
            expected={"contains": ["variance", "budget", "IFRS 15"], "score_min": 0.7},
            metadata={"scenario": "revenue_recognition", "priority": "high"},
        )
        self.add_case(ds,
            input={"task": "Identify anomalies in accounts payable ledger", "period": "2025-09"},
            expected={"contains": ["anomaly", "threshold", "confidence"], "score_min": 0.7},
            metadata={"scenario": "anomaly_detection"},
        )
        self.add_case(ds,
            input={"task": "Reconcile cash position as of month-end", "period": "2025-09"},
            expected={"contains": ["reconciliation", "balance", "difference"], "score_min": 0.75},
            metadata={"scenario": "reconciliation"},
        )
        self.add_case(ds,
            input={"task": "Assess financial risk for Q3 2025", "period": "2025-09"},
            expected={"contains": ["risk", "mitigation", "score"], "score_min": 0.7},
            metadata={"scenario": "risk_assessment"},
        )

    def get_dataset(self, name: str) -> List[EvalCase]:
        return self._datasets.get(name, [])

    # ── Run evaluation ────────────────────────────────────────────────────────

    async def evaluate(
        self,
        run_fn: Callable,
        dataset: str,
        evaluators: Optional[List[str]] = None,
    ) -> EvalReport:
        """
        Run `run_fn` on every case in `dataset` and score with LLM judges.

        Args:
            run_fn:     async callable (input_dict) → output (str or dict)
            dataset:    name of the dataset to run
            evaluators: list of evaluator names (default: all 6)
        """
        evaluators = evaluators or list(_EVAL_PROMPTS.keys())
        cases = self.get_dataset(dataset)
        if not cases:
            logger.warning("Dataset '%s' is empty. Add cases first.", dataset)
            return EvalReport(dataset=dataset, results=[], pass_threshold=self._settings.eval_pass_threshold)

        logger.info("Evaluating %d cases from dataset '%s'", len(cases), dataset)

        sem = asyncio.Semaphore(self._settings.eval_max_concurrency)

        async def eval_one(case: EvalCase) -> EvalResult:
            async with sem:
                return await self._run_one(case, run_fn, evaluators)

        results = await asyncio.gather(*[eval_one(c) for c in cases], return_exceptions=True)

        processed: List[EvalResult] = []
        for case, result in zip(cases, results):
            if isinstance(result, Exception):
                processed.append(EvalResult(
                    case=case, output=None, scores=[],
                    error=str(result),
                ))
            else:
                processed.append(result)

        report = EvalReport(
            dataset=dataset,
            results=processed,
            pass_threshold=self._settings.eval_pass_threshold,
        )
        self._history.append(report)
        logger.info("Eval complete: avg=%.2f pass_rate=%.0f%%", report.avg_score, report.pass_rate * 100)
        return report

    async def _run_one(
        self,
        case: EvalCase,
        run_fn: Callable,
        evaluators: List[str],
    ) -> EvalResult:
        start = time.monotonic()
        output = None
        error = None

        try:
            output = await run_fn(case.input)
        except Exception as exc:
            error = str(exc)
            logger.warning("run_fn failed for case %s: %s", case.case_id, exc)

        latency_ms = (time.monotonic() - start) * 1000

        scores: List[EvalScore] = []

        # Rule-based checks (no LLM needed) — fast and cheap
        scores.extend(self._rule_based_checks(case, output))

        # LLM-as-judge for each evaluator
        if output is not None:
            judge_tasks = [
                self._llm_judge(ev, case, output)
                for ev in evaluators
                if ev in _EVAL_PROMPTS
            ]
            judge_scores = await asyncio.gather(*judge_tasks, return_exceptions=True)
            for score_or_exc in judge_scores:
                if isinstance(score_or_exc, EvalScore):
                    scores.append(score_or_exc)

        return EvalResult(case=case, output=output, scores=scores, latency_ms=latency_ms, error=error)

    def _rule_based_checks(self, case: EvalCase, output: Any) -> List[EvalScore]:
        """Fast deterministic checks — run before expensive LLM judges."""
        scores: List[EvalScore] = []
        expected = case.expected
        output_str = json.dumps(output) if not isinstance(output, str) else output

        # Keyword presence check
        if "contains" in expected:
            keywords = expected["contains"]
            hits = sum(1 for kw in keywords if kw.lower() in output_str.lower())
            score = hits / len(keywords) if keywords else 1.0
            scores.append(EvalScore(
                evaluator="keyword_presence",
                score=score,
                reasoning=f"{hits}/{len(keywords)} expected keywords found",
            ))

        # Minimum score gate (from metadata)
        min_score = expected.get("score_min")
        if min_score and not output:
            scores.append(EvalScore(
                evaluator="output_exists",
                score=0.0,
                reasoning="No output produced",
            ))

        return scores

    async def _llm_judge(
        self, evaluator: str, case: EvalCase, output: Any
    ) -> EvalScore:
        prompt_template = _EVAL_PROMPTS.get(evaluator, "")
        if not prompt_template:
            return EvalScore(evaluator=evaluator, score=0.5, reasoning="Unknown evaluator")

        output_str = json.dumps(output, default=str) if not isinstance(output, str) else output
        prompt = prompt_template.format(
            input=json.dumps(case.input, default=str),
            output=output_str[:2000],
            expected=json.dumps(case.expected, default=str),
        )

        try:
            response = await self._llm.ainvoke([
                SystemMessage(content=_JUDGE_SYSTEM),
                HumanMessage(content=prompt),
            ])
            content = response.content.strip()
            # Extract JSON from possible markdown fences
            if "```" in content:
                content = content.split("```")[1].lstrip("json\n")
            parsed = json.loads(content)
            return EvalScore(
                evaluator=evaluator,
                score=float(parsed.get("score", 0.5)),
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception as exc:
            logger.debug("LLM judge failed (%s): %s", evaluator, exc)
            return EvalScore(evaluator=evaluator, score=0.5, reasoning=f"Judge error: {exc}")

    # ── History & regression ─────────────────────────────────────────────────

    def compare_reports(self, baseline: EvalReport, current: EvalReport) -> str:
        """Show score delta between two eval runs."""
        delta = current.avg_score - baseline.avg_score
        direction = "▲" if delta >= 0 else "▼"
        return (
            f"Regression check: {baseline.dataset}\n"
            f"  Baseline avg:  {baseline.avg_score:.2f}\n"
            f"  Current avg:   {current.avg_score:.2f}\n"
            f"  Delta:         {direction} {abs(delta):.2f}\n"
            f"  Status:        {'✅ No regression' if delta >= -0.05 else '❌ Regression detected'}"
        )
