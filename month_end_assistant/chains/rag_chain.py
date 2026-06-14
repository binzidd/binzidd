"""
RAG (Retrieval-Augmented Generation) Chain for Accounting Standards.

Builds an in-memory vector store populated with IFRS / US GAAP guidance
snippets and exposes a retrieval-augmented Q&A chain for:

  • Policy lookup ("What does IFRS 15 say about variable consideration?")
  • Compliance checking ("Does our revenue recognition comply with ASC 606?")
  • Month-end checklist generation ("What must we do before closing under IAS 2?")

LangChain features showcased:
  • InMemoryVectorStore      – zero-dependency vector store (swap for Pinecone etc.)
  • FakeEmbeddings           – deterministic embeddings for demo / CI (swap for BedrockEmbeddings)
  • RecursiveCharacterTextSplitter – chunk documents for indexing
  • create_retrieval_chain   – LCEL retrieval chain factory
  • create_stuff_documents_chain  – simple document stuffing strategy
  • RunnablePassthrough      – passes question through to the final answer
  • ChatPromptTemplate       – custom RAG prompt with context injection
  • with_config              – per-invocation config override (callbacks, tags)
  • Document                 – LangChain document model
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Accounting standards knowledge base
# ─────────────────────────────────────────────────────────────────────────────

_STANDARDS_CORPUS: List[Dict[str, str]] = [
    # ── Revenue ──────────────────────────────────────────────────────────────
    {
        "title": "IFRS 15 / ASC 606 – Revenue Recognition Five-Step Model",
        "content": (
            "IFRS 15 and ASC 606 establish a five-step model for revenue recognition: "
            "(1) Identify the contract(s) with a customer. "
            "(2) Identify the performance obligations in the contract. "
            "(3) Determine the transaction price, including variable consideration "
            "(estimates constrained to avoid significant revenue reversal). "
            "(4) Allocate the transaction price to performance obligations using "
            "standalone selling prices. "
            "(5) Recognise revenue when (or as) each performance obligation is satisfied. "
            "For point-in-time recognition, control must transfer to the customer. "
            "For over-time recognition, one of three criteria must be met: "
            "the customer simultaneously receives and consumes the benefit, "
            "the entity creates or enhances an asset the customer controls, or "
            "the asset has no alternative use and the entity has an enforceable right to payment."
        ),
        "standard": "IFRS 15 / ASC 606",
        "topic": "revenue recognition",
    },
    {
        "title": "IFRS 15 – Contract Modifications",
        "content": (
            "A contract modification is a change in the scope or price of a contract. "
            "If the modification adds distinct goods or services at standalone selling price, "
            "account for it as a separate contract. "
            "Otherwise, account for it as a termination of the existing contract and creation "
            "of a new one (if remaining goods are distinct) or as part of the existing contract "
            "(if remaining goods are not distinct), using the catch-up method."
        ),
        "standard": "IFRS 15",
        "topic": "revenue recognition contract modifications",
    },
    # ── Inventory ─────────────────────────────────────────────────────────────
    {
        "title": "IAS 2 / ASC 330 – Inventory Valuation",
        "content": (
            "Inventories must be measured at the lower of cost and net realisable value (NRV). "
            "Cost includes purchase price, conversion costs, and other directly attributable costs. "
            "IFRS prohibits LIFO; permitted cost formulas are FIFO and weighted average. "
            "US GAAP permits LIFO, FIFO, and weighted average. "
            "NRV is the estimated selling price less estimated costs to complete and sell. "
            "Write-downs to NRV are recognised in profit or loss in the period they occur. "
            "Month-end close procedures: perform physical count or cycle count, "
            "adjust perpetual records, test for NRV write-down triggers."
        ),
        "standard": "IAS 2 / ASC 330",
        "topic": "inventory valuation month-end close",
    },
    # ── Leases ────────────────────────────────────────────────────────────────
    {
        "title": "IFRS 16 / ASC 842 – Lease Accounting",
        "content": (
            "IFRS 16 requires lessees to recognise a right-of-use (ROU) asset and a lease liability "
            "for all leases with a term over 12 months and that are not low-value. "
            "The lease liability is initially measured at the present value of lease payments "
            "discounted at the incremental borrowing rate (or implicit rate if known). "
            "Subsequently, the liability is reduced by principal repayments and increased by "
            "interest accretion. "
            "Month-end journal entries: Dr Interest Expense / Cr Lease Liability (interest), "
            "Dr Depreciation / Cr Accumulated Depreciation (ROU asset depreciation). "
            "ASC 842 uses a similar dual-model approach (finance leases vs operating leases). "
            "Operating leases under ASC 842 recognise a straight-line lease cost."
        ),
        "standard": "IFRS 16 / ASC 842",
        "topic": "lease accounting month-end journal entries",
    },
    # ── Impairment ────────────────────────────────────────────────────────────
    {
        "title": "IAS 36 – Impairment of Assets",
        "content": (
            "Assets must be tested for impairment when impairment indicators exist. "
            "Indicators include: significant decline in market value, adverse changes in "
            "the technological, market, economic, or legal environment, and internal evidence "
            "of obsolescence or physical damage. "
            "Goodwill and intangibles with indefinite useful lives are tested annually regardless. "
            "The recoverable amount is the higher of: fair value less costs of disposal (FVLCD) "
            "and value in use (VIU, the discounted future cash flows). "
            "If the carrying amount exceeds the recoverable amount, an impairment loss is "
            "recognised in profit or loss. Impairment losses on goodwill cannot be reversed."
        ),
        "standard": "IAS 36",
        "topic": "impairment testing goodwill intangibles",
    },
    # ── Provisions ────────────────────────────────────────────────────────────
    {
        "title": "IAS 37 / ASC 450 – Provisions and Contingencies",
        "content": (
            "A provision is recognised when: "
            "(1) there is a present obligation (legal or constructive) as a result of a past event, "
            "(2) it is probable that an outflow of resources will be required, "
            "(3) a reliable estimate can be made of the amount. "
            "Contingent liabilities are disclosed but not recognised unless the transfer of "
            "economic benefits is probable. "
            "Month-end close: review open litigation, warranties, restructuring plans, and "
            "environmental obligations. "
            "Under US GAAP (ASC 450), 'probable' and 'reasonably estimable' triggers recognition."
        ),
        "standard": "IAS 37 / ASC 450",
        "topic": "provisions contingencies month-end accruals",
    },
    # ── Cash flow ─────────────────────────────────────────────────────────────
    {
        "title": "IAS 7 / ASC 230 – Statement of Cash Flows",
        "content": (
            "Cash flows are classified into operating, investing, and financing activities. "
            "Operating activities can be presented using the direct or indirect method. "
            "Under the indirect method, profit is adjusted for: "
            "non-cash items (depreciation, amortisation, impairment), "
            "changes in working capital (trade receivables, inventories, payables), "
            "and items to be classified as investing or financing. "
            "Month-end close procedures include bank reconciliation, confirmation of outstanding "
            "cheques and deposits in transit, and review of restricted cash disclosures."
        ),
        "standard": "IAS 7 / ASC 230",
        "topic": "cash flow statement bank reconciliation month-end",
    },
    # ── Deferred tax ─────────────────────────────────────────────────────────
    {
        "title": "IAS 12 / ASC 740 – Income Taxes (Deferred Tax)",
        "content": (
            "Deferred tax liabilities are recognised for taxable temporary differences; "
            "deferred tax assets for deductible temporary differences (to the extent probable "
            "that sufficient future taxable profit will be available). "
            "Common month-end items: depreciation timing differences, warranty provisions, "
            "share-based payments, and lease liabilities under IFRS 16. "
            "Measurement: use the enacted (or substantively enacted) tax rate at the reporting date. "
            "US GAAP (ASC 740) requires a valuation allowance instead of the IFRS probability test."
        ),
        "standard": "IAS 12 / ASC 740",
        "topic": "deferred tax income taxes month-end",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RAG Chain class
# ─────────────────────────────────────────────────────────────────────────────

class AccountingRAGChain:
    """
    RAG chain that answers accounting-standards questions using an in-memory
    vector store pre-populated with IFRS / US GAAP guidance.

    Swap FakeEmbeddings for BedrockEmbeddings in production:
        from langchain_aws import BedrockEmbeddings
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    Usage:
        rag   = AccountingRAGChain(llm)
        reply = rag.ask("What journal entries are needed for IFRS 16 month-end?")
    """

    # Chunk size / overlap for the text splitter
    _CHUNK_SIZE    = 500
    _CHUNK_OVERLAP = 80

    def __init__(self, llm: Any, embeddings: Optional[Any] = None) -> None:
        self._llm        = llm
        self._embeddings = embeddings or FakeEmbeddings(size=1536)
        self._vectorstore = self._build_vectorstore()
        self._chain       = self._build_chain()
        logger.info("AccountingRAGChain ready with %d documents.", len(_STANDARDS_CORPUS))

    # ── Vector store ──────────────────────────────────────────────────────────

    def _build_vectorstore(self) -> InMemoryVectorStore:
        """
        Split corpus into chunks and index them in an InMemoryVectorStore.

        InMemoryVectorStore stores vectors in a Python dict – zero latency,
        zero infrastructure.  For production, swap in:
            Amazon OpenSearch, Pinecone, Chroma, pgvector, etc.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=self._CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

        documents: List[Document] = []
        for item in _STANDARDS_CORPUS:
            chunks = splitter.split_text(item["content"])
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "title":    item["title"],
                        "standard": item["standard"],
                        "topic":    item["topic"],
                        "chunk_id": i,
                    },
                ))

        vectorstore = InMemoryVectorStore(embedding=self._embeddings)
        vectorstore.add_documents(documents)
        return vectorstore

    # ── Chain construction ────────────────────────────────────────────────────

    def _build_chain(self) -> Any:
        """
        Build the LCEL retrieval chain using the stuff-documents strategy.

        Chain flow:
            question
              ↓  retriever (semantic search → top-k docs)
            {context: [doc1, doc2…], input: question}
              ↓  stuff_chain (prompt + LLM + StrOutputParser)
            answer string
        """
        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert in IFRS and US GAAP accounting standards. "
                "Answer the question using ONLY the provided context. "
                "If the context does not contain enough information, say so explicitly. "
                "Cite the relevant standard (e.g. 'IFRS 15 §47') when possible.\n\n"
                "Context:\n{context}"
            )),
            ("human", "{input}"),
        ])

        # create_stuff_documents_chain formats retrieved docs into the prompt
        stuff_chain = create_stuff_documents_chain(self._llm, rag_prompt)

        # create_retrieval_chain wires retriever → stuff_chain
        return create_retrieval_chain(retriever, stuff_chain)

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(self, question: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Answer an accounting-standards question using RAG.

        Args:
            question: Natural language question about IFRS / GAAP.
            tags:     Optional LangSmith tags for tracing.

        Returns:
            Dict with 'answer' (str) and 'source_documents' (list of Document).
        """
        config = {}
        if tags:
            config["tags"] = tags

        result = self._chain.invoke(
            {"input": question},
            config=config or None,
        )
        return {
            "answer":           result.get("answer", ""),
            "source_documents": result.get("context", []),
            "question":         question,
        }

    async def aask(self, question: str) -> Dict[str, Any]:
        """Async version of ask() for use inside async agent nodes."""
        result = await self._chain.ainvoke({"input": question})
        return {
            "answer":           result.get("answer", ""),
            "source_documents": result.get("context", []),
            "question":         question,
        }

    def stream_answer(self, question: str):
        """
        Stream the answer token-by-token using LCEL's built-in streaming.

        Usage (in a FastAPI SSE endpoint):
            for chunk in rag.stream_answer(question):
                yield f"data: {chunk}\\n\\n"
        """
        for chunk in self._chain.stream({"input": question}):
            answer_chunk = chunk.get("answer", "")
            if answer_chunk:
                yield answer_chunk

    def add_custom_policy(self, title: str, content: str, tags: List[str]) -> None:
        """
        Add a company-specific accounting policy to the knowledge base at runtime.

        Useful for loading custom accounting memos or policy documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=self._CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(content)
        docs   = [
            Document(
                page_content=chunk,
                metadata={"title": title, "standard": "company-policy", "topic": " ".join(tags)},
            )
            for chunk in chunks
        ]
        self._vectorstore.add_documents(docs)
        logger.info("Added custom policy '%s' (%d chunks)", title, len(docs))
