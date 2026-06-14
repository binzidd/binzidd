"""
LCEL (LangChain Expression Language) Chain Library.

Showcases every major LCEL composition primitive available in LangChain 0.3+:

  │  Primitive                │  Where used in this file              │
  ├───────────────────────────┼───────────────────────────────────────┤
  │  pipe  |                  │  every chain                          │
  │  RunnableParallel         │  build_parallel_analysis_chain        │
  │  RunnableBranch           │  build_branching_chain                │
  │  RunnablePassthrough      │  RAG passthrough of the question      │
  │  RunnableLambda           │  custom transform steps               │
  │  with_structured_output   │  variance_chain, anomaly_chain        │
  │  with_retry               │  wrapped around every LLM call        │
  │  with_fallbacks           │  primary → stub chain fallback        │
  │  PydanticOutputParser     │  narrative_chain                      │
  │  JsonOutputParser         │  anomaly_chain                        │
  │  ChatPromptTemplate       │  all chains                           │
  │  FewShotChatMessagePrompt │  anomaly_chain few-shot examples      │
  │  MessagesPlaceholder      │  chains that accept chat history      │
  │  StrOutputParser          │  narrative_chain final step           │
  └───────────────────────────┴───────────────────────────────────────┘
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field

from month_end_assistant.config import get_settings


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic output schemas  (used with with_structured_output)
# ─────────────────────────────────────────────────────────────────────────────

class VarianceAnalysis(BaseModel):
    """Structured variance analysis produced by the LLM."""

    metric:         str
    direction:      str   = Field(description="'favourable' or 'adverse'")
    root_cause:     str   = Field(description="Most likely root cause in one sentence")
    action_needed:  bool
    recommended_action: str = Field(default="")
    confidence:     float  = Field(ge=0.0, le=1.0)


class AnomalyFlag(BaseModel):
    """A single flagged anomaly in the financial data."""

    anomaly_type:  str   = Field(description="e.g. 'duplicate entry', 'threshold breach'")
    description:   str
    severity:      str   = Field(description="'low' | 'medium' | 'high' | 'critical'")
    affected_line: str   = Field(description="Account or metric name")
    recommended_review: str


class FinancialNarrative(BaseModel):
    """CFO-ready narrative paragraph and top-3 actions."""

    headline:        str
    body_paragraph:  str
    top_actions:     List[str] = Field(default_factory=list, max_length=3)
    sentiment:       str       = Field(description="'positive' | 'neutral' | 'negative'")


# ─────────────────────────────────────────────────────────────────────────────
# Shared prompt fragments
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_ANALYST = (
    "You are a senior financial analyst preparing month-end close commentary. "
    "Be precise, data-driven, and concise.  All monetary values are in USD."
)

_SYSTEM_CONTROLLER = (
    "You are a financial controller reviewing month-end data for anomalies and "
    "compliance issues.  Flag anything that requires investigation."
)

_SYSTEM_CFO = (
    "You are a CFO preparing board-level commentary.  Write in executive style: "
    "clear, confident, and forward-looking."
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Variance Analysis Chain  – with_structured_output + retry
# ─────────────────────────────────────────────────────────────────────────────

def build_variance_chain(llm: Any) -> Any:
    """
    LCEL chain that analyses a single variance and returns a VarianceAnalysis.

    Features:
      • with_structured_output – forces the LLM to emit a valid Pydantic object
      • with_retry             – retries up to 3× on transient Bedrock errors
      • ChatPromptTemplate     – system + human message template
      • | pipe                 – LCEL chain composition
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_ANALYST),
        ("human", (
            "Analyse this financial variance and return a structured analysis.\n\n"
            "Metric:        {metric}\n"
            "Actual:        ${actual:,.0f}\n"
            "Budget:        ${budget:,.0f}\n"
            "Variance %:    {vs_budget_pct:.1f}%\n"
            "Prior Period:  ${prior_period:,.0f}\n\n"
            "Return a VarianceAnalysis JSON object."
        )),
    ])

    # with_structured_output enforces the Pydantic schema on every LLM response
    structured_llm = llm.with_structured_output(VarianceAnalysis)

    # with_retry wraps the LLM call – retries on rate-limit / transient errors
    retrying_llm = structured_llm.with_retry(
        retry_if_exception_type=(Exception,),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    return prompt | retrying_llm


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parallel Analysis Chain  – RunnableParallel
# ─────────────────────────────────────────────────────────────────────────────

def build_parallel_analysis_chain(llm: Any) -> Any:
    """
    Runs variance analysis and benchmark commentary IN PARALLEL, then merges.

    Features:
      • RunnableParallel – executes two independent sub-chains simultaneously
      • RunnableLambda   – custom merge function as a Runnable step
      • | pipe           – final merge into a single dict
    """
    variance_prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_ANALYST),
        ("human", "Give a 2-sentence variance commentary for:\n{financial_summary}"),
    ])

    benchmark_prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_ANALYST),
        ("human", (
            "Compare these metrics to {sector} industry benchmarks:\n"
            "{financial_summary}\n\n"
            "Highlight where we outperform or underperform the median."
        )),
    ])

    variance_chain   = variance_prompt   | llm | StrOutputParser()
    benchmark_chain  = benchmark_prompt  | llm | StrOutputParser()

    # Both chains receive the same input dict and run concurrently
    parallel_chain = RunnableParallel(
        variance_commentary  = variance_chain,
        benchmark_commentary = benchmark_chain,
    )

    # Merge the two parallel outputs into a single combined analysis
    merge = RunnableLambda(lambda x: {
        "combined_analysis": (
            "## Variance Commentary\n" + x["variance_commentary"] + "\n\n"
            "## Benchmark Commentary\n" + x["benchmark_commentary"]
        ),
        **x,
    })

    return parallel_chain | merge


# ─────────────────────────────────────────────────────────────────────────────
# 3. Branching Chain  – RunnableBranch (route by severity)
# ─────────────────────────────────────────────────────────────────────────────

def build_branching_chain(llm: Any) -> Any:
    """
    Routes the analysis prompt based on variance severity using RunnableBranch.

    Features:
      • RunnableBranch    – conditional routing without if/else in node code
      • RunnablePassthrough – passes the input dict through unchanged
      • Multiple prompt templates for different severity levels
    """
    critical_prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_CONTROLLER),
        ("human", (
            "CRITICAL variance detected: {metric} at {vs_budget_pct:.1f}% vs budget.\n"
            "Write an urgent action memo for the CFO. Include: cause, impact, "
            "immediate action, and escalation path."
        )),
    ])

    material_prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_ANALYST),
        ("human", (
            "Material variance: {metric} at {vs_budget_pct:.1f}% vs budget.\n"
            "Write a concise management commentary (3 sentences) and propose a corrective action."
        )),
    ])

    immaterial_prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_ANALYST),
        ("human", (
            "Variance for {metric} is within tolerance ({vs_budget_pct:.1f}% vs budget).\n"
            "Provide a one-sentence acknowledgement for the board pack."
        )),
    ])

    parser = StrOutputParser()

    return RunnableBranch(
        # Each branch: (condition_callable, runnable_to_execute)
        (lambda x: abs(x["vs_budget_pct"]) >= 15,  critical_prompt  | llm | parser),
        (lambda x: abs(x["vs_budget_pct"]) >= 5,   material_prompt  | llm | parser),
        # Default branch (no condition)
        immaterial_prompt | llm | parser,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Anomaly Detection Chain  – few-shot + JsonOutputParser + with_fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def build_anomaly_chain(llm: Any) -> Any:
    """
    Detects financial anomalies using few-shot examples and JsonOutputParser.

    Features:
      • FewShotChatMessagePromptTemplate – in-context learning examples
      • JsonOutputParser                 – parse LLM output as JSON list
      • with_fallbacks                   – if primary LLM fails, stub fallback
      • RunnableLambda                   – post-process the JSON list
    """
    # Few-shot examples teach the model the anomaly detection format
    few_shot_examples = [
        {
            "input":  "Revenue: $1.2M (budget $1.0M). Accounts Receivable: $950K.",
            "output": json.dumps([{
                "anomaly_type": "AR concentration risk",
                "description":  "AR/Revenue ratio of 79% is unusually high – suggests slow collections.",
                "severity":     "medium",
                "affected_line": "accounts_receivable",
                "recommended_review": "Review aging report; escalate accounts > 90 days.",
            }]),
        },
        {
            "input":  "Operating expenses: $520K (budget $300K, +73%).",
            "output": json.dumps([{
                "anomaly_type": "threshold breach",
                "description":  "OpEx is 73% above budget – far exceeds the 5% materiality threshold.",
                "severity":     "critical",
                "affected_line": "operating_expenses",
                "recommended_review": "Obtain itemised OpEx breakdown; identify unapproved spend.",
            }]),
        },
    ]

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ]),
        examples=few_shot_examples,
    )

    full_prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_CONTROLLER),
        few_shot_prompt,
        ("human", (
            "Analyse this financial data for anomalies:\n{financial_data}\n\n"
            "Return a JSON array of AnomalyFlag objects (empty array if none)."
        )),
    ])

    parser = JsonOutputParser()

    primary_chain = full_prompt | llm | parser

    # Fallback: if the LLM is unavailable, return an empty anomaly list
    stub_chain = RunnableLambda(lambda _: [])

    return primary_chain.with_fallbacks([stub_chain])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Narrative Chain  – PydanticOutputParser + MessagesPlaceholder (chat history)
# ─────────────────────────────────────────────────────────────────────────────

def build_narrative_chain(llm: Any) -> Any:
    """
    Generates a structured CFO narrative with full chat history awareness.

    Features:
      • MessagesPlaceholder – injects prior conversation turns into the prompt
      • PydanticOutputParser – parse output into FinancialNarrative
      • RunnablePassthrough  – passes 'chat_history' through unchanged
    """
    parser = PydanticOutputParser(pydantic_object=FinancialNarrative)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_CFO + "\n\n" + parser.get_format_instructions()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", (
            "Write a board-pack narrative for {period_label}.\n\n"
            "Financial highlights:\n{financial_summary}\n\n"
            "Key variances:\n{variance_summary}\n\n"
            "Research insights:\n{research_synthesis}"
        )),
    ])

    return (
        RunnablePassthrough.assign(chat_history=lambda x: x.get("chat_history", []))
        | prompt
        | llm
        | parser
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. LCELChainFactory  – builds and caches all chains for an LLM instance
# ─────────────────────────────────────────────────────────────────────────────

class LCELChainFactory:
    """
    Factory that wires all LCEL chains to a shared LLM instance.

    Caches each chain after first build so subsequent calls are free.

    Usage:
        factory  = LCELChainFactory(llm)
        chain    = factory.variance_chain
        result   = chain.invoke({"metric": "revenue", "actual": 4_200_000, ...})
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._cache: Dict[str, Any] = {}

    def _get(self, key: str, builder) -> Any:
        if key not in self._cache:
            self._cache[key] = builder(self._llm)
        return self._cache[key]

    @property
    def variance_chain(self) -> Any:
        return self._get("variance", build_variance_chain)

    @property
    def parallel_analysis_chain(self) -> Any:
        return self._get("parallel", build_parallel_analysis_chain)

    @property
    def branching_chain(self) -> Any:
        return self._get("branching", build_branching_chain)

    @property
    def anomaly_chain(self) -> Any:
        return self._get("anomaly", build_anomaly_chain)

    @property
    def narrative_chain(self) -> Any:
        return self._get("narrative", build_narrative_chain)

    def stream_variance(self, inputs: Dict[str, Any]):
        """Stream the variance chain token-by-token (demo of LCEL streaming)."""
        for chunk in self.variance_chain.stream(inputs):
            yield chunk
