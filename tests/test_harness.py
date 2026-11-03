"""Comprehensive test suite for the standalone RAG Evaluation Harness."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rag_harness.adapters import CallablePipelineAdapter, TraceFileAdapter
from rag_harness.cli import app
from rag_harness.demos.corpus import BENCHMARK_QUESTIONS
from rag_harness.demos.public_repos import (
    CorrectiveSelfRAGPipeline,
    HybridBM25RerankRAGPipeline,
    NaiveChromaRAGPipeline,
)
from rag_harness.evaluator import RAGEvaluator
from rag_harness.metrics.claims import decompose_into_claims, is_abstention_response
from rag_harness.metrics.ensemble import DualFaithfulnessEngine
from rag_harness.metrics.nli_judge import NLIEvaluator
from rag_harness.metrics.retrieval import evaluate_retrieval
from rag_harness.schemas import (
    ClaimVerdict,
    FailureCategory,
    NLILabel,
    QuestionItem,
    RetrievedChunk,
)


def test_retrieval_metrics_calculation() -> None:
    q = QuestionItem(
        query_id="test_q1",
        question="What is the refund policy?",
        ground_truth_answer="Standard plans have a 30-day refund window.",
        golden_chunk_ids=["doc_a", "doc_b"],
        key_facts=["30-day", "refund"],
    )
    chunks = [
        RetrievedChunk(chunk_id="doc_x", text="Unrelated text.", score=0.9, rank=1),
        RetrievedChunk(chunk_id="doc_a", text="Standard plans have a 30-day refund window.", score=0.8, rank=2),
        RetrievedChunk(chunk_id="doc_y", text="More unrelated text.", score=0.7, rank=3),
    ]
    metrics = evaluate_retrieval(q, chunks, k=3)
    assert metrics.recall_at_k == 0.5
    assert round(metrics.precision_at_k, 4) == round(1 / 3, 4)
    assert metrics.mrr == 0.5  # first relevant hit at rank 2 -> 1/2 = 0.5
    assert metrics.first_relevant_rank == 2
    assert metrics.hit_rate == 1.0
    assert metrics.context_fact_coverage == 1.0


def test_claim_decomposition_and_abstention() -> None:
    assert is_abstention_response("I do not have enough information in the provided context to answer.")
    assert not is_abstention_response("Standard plans allow 120 requests per minute.")

    claims = decompose_into_claims(
        "Standard plans allow 120 requests per minute, while Enterprise plans support 5,000 requests per minute. "
        "Exceeding the limit returns HTTP 429."
    )
    assert len(claims) == 3
    assert "120 requests per minute" in claims[0]
    assert "5,000 requests per minute" in claims[1]


def test_nli_catches_entity_tier_swap_where_llm_judge_agrees() -> None:
    chunk = RetrievedChunk(
        chunk_id="doc_01",
        text=(
            "Standard tier subscriptions include a 30-day full refund window from the initial purchase date. "
            "Enterprise tier annual contracts are strictly non-refundable once provisioned."
        ),
        score=0.95,
        rank=1,
    )
    # Subtle hallucination: attributes Standard tier's 30-day refund window to Enterprise tier
    tricky_answer = "Enterprise tier annual contracts include a 30-day full refund window from the initial purchase date."

    engine = DualFaithfulnessEngine()
    res = engine.evaluate(tricky_answer, [chunk])

    assert len(res.claims) == 1
    claim_eval = res.claims[0]
    assert claim_eval.llm_verdict == ClaimVerdict.SUPPORTED
    assert claim_eval.nli_label == NLILabel.CONTRADICTION
    assert not claim_eval.agreement
    assert claim_eval.disagreement_type == "LLM_MISSED_CONTRADICTION"
    assert res.has_judge_disagreement
    assert res.cross_checked_score < 0.15


def test_nli_entailment_on_faithful_answer() -> None:
    chunk = RetrievedChunk(
        chunk_id="doc_04",
        text="All customer vector indexes are encrypted at rest using AES-256-GCM with keys rotated automatically every 90 days.",
        score=0.95,
        rank=1,
    )
    faithful_answer = "Customer vector indexes are encrypted at rest using AES-256-GCM and keys are rotated every 90 days."
    nli = NLIEvaluator()
    out = nli.score_claim(faithful_answer, [chunk])
    assert out.label == NLILabel.ENTAILMENT
    assert out.probabilities.entailment > 0.80


def test_three_public_repo_benchmarks_progression(tmp_path: Path) -> None:
    evaluator = RAGEvaluator(k=5, cache_path=tmp_path / "cache.sqlite")

    naive_adapter = CallablePipelineAdapter(NaiveChromaRAGPipeline().query, "langchain-naive-chroma-rag")
    hybrid_adapter = CallablePipelineAdapter(HybridBM25RerankRAGPipeline().query, "llamaindex-hybrid-bm25-rerank")
    corrective_adapter = CallablePipelineAdapter(CorrectiveSelfRAGPipeline().query, "langgraph-corrective-self-rag")

    trace_file = tmp_path / "naive.traces.jsonl"
    r_naive = evaluator.run(naive_adapter, BENCHMARK_QUESTIONS, trace_log_path=trace_file)
    r_hybrid = evaluator.run(hybrid_adapter, BENCHMARK_QUESTIONS)
    r_corrective = evaluator.run(corrective_adapter, BENCHMARK_QUESTIONS)

    # Verify progressive improvement across the 3 architectures
    assert r_naive.pass_rate < r_hybrid.pass_rate < r_corrective.pass_rate
    assert r_naive.mean_recall_at_k < r_hybrid.mean_recall_at_k
    assert r_naive.mean_cross_checked_faithfulness < r_hybrid.mean_cross_checked_faithfulness < r_corrective.mean_cross_checked_faithfulness
    assert r_naive.total_judge_disagreements > 0
    assert len(r_naive.representative_failures) >= 4

    # Verify TraceFileAdapter reproduces identical scores from saved JSONL trace file
    trace_adapter = TraceFileAdapter(trace_file)
    r_from_traces = evaluator.run(trace_adapter, BENCHMARK_QUESTIONS)
    assert r_from_traces.pass_rate == r_naive.pass_rate
    assert r_from_traces.mean_cross_checked_faithfulness == r_naive.mean_cross_checked_faithfulness


def test_cli_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    out_html = tmp_path / "report.html"

    res = runner.invoke(
        app,
        [
            "run",
            "--pipeline",
            "langgraph-corrective-self-rag",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-html",
            str(out_html),
            "--fail-under-faithfulness",
            "0.90",
            "--fail-under-recall",
            "0.90",
        ],
    )
    assert res.exit_code == 0
    assert out_json.exists()
    assert out_md.exists()
    assert out_html.exists()
