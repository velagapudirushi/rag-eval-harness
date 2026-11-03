"""Interactive Streamlit Web UI for `rag-eval-harness` (1-click deployable from GitHub)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure local `src/` is on path when deployed on Streamlit Community Cloud directly from GitHub
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_harness.adapters import CallablePipelineAdapter
from rag_harness.demos.corpus import BENCHMARK_QUESTIONS
from rag_harness.demos.public_repos import PUBLIC_REPO_PIPELINES
from rag_harness.evaluator import RAGEvaluator
from rag_harness.metrics.ensemble import DualFaithfulnessEngine
from rag_harness.schemas import QuestionItem, RAGTrace, RetrievedChunk

st.set_page_config(
    page_title="RAG Eval Harness — Dual-Judge Benchmark & Triage",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Standalone RAG Evaluation Harness")
st.caption(
    "Framework-agnostic CLI & Python library: **Retrieval Metrics (Recall@K, MRR, NDCG@K)** • "
    "**Dual-Engine Faithfulness (LLM-as-a-Judge × NLI Entailment Cross-Check)** • **Representative Failure Triage**"
)


@st.cache_data
def run_cached_benchmarks():
    evaluator = RAGEvaluator(k=5)
    reports = {}
    for name, cls in PUBLIC_REPO_PIPELINES.items():
        inst = cls()
        adapter = CallablePipelineAdapter(inst.query, pipeline_name=name)
        reports[name] = evaluator.run(adapter, BENCHMARK_QUESTIONS)
    return reports


tab_bench, tab_sandbox, tab_upload = st.tabs(
    [
        "📊 3-Repo Public Benchmark Demo",
        "⚡ Live Dual-Judge Faithfulness Inspector",
        "📁 Evaluate Custom JSONL Traces",
    ]
)

with tab_bench:
    reports = run_cached_benchmarks()
    summary_rows = []
    for name, r in reports.items():
        summary_rows.append(
            {
                "RAG Pipeline / Repo": name,
                "Pass Rate": f"{r.pass_rate:.1%} ({r.passed_queries}/{r.total_queries})",
                "Recall@5": f"{r.mean_recall_at_k:.1%}",
                "MRR": round(r.mean_mrr, 3),
                "NDCG@5": round(r.mean_ndcg_at_k, 3),
                "LLM Judge Only": f"{r.mean_llm_faithfulness:.1%}",
                "NLI Entailment": f"{r.mean_nli_faithfulness:.1%}",
                "Cross-Checked Score": f"{r.mean_cross_checked_faithfulness:.1%}",
                "Judge-NLI Disagreements": f"{r.total_judge_disagreements}/{r.total_claims_evaluated} ({r.disagreement_rate:.1%})",
                "Mean Latency (ms)": r.mean_latency_ms,
            }
        )
    st.subheader("1. Head-to-Head RAG Architecture Comparison")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    selected_repo = st.selectbox(
        "Inspect Detailed Failure Report & Traces for:",
        options=list(reports.keys()),
        index=0,
    )
    rep = reports[selected_repo]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pass Rate", f"{rep.pass_rate:.1%}", f"{rep.passed_queries}/{rep.total_queries} queries")
    c2.metric("Recall@5 / MRR", f"{rep.mean_recall_at_k:.1%}", f"MRR: {rep.mean_mrr:.3f}")
    c3.metric(
        "Cross-Checked Faithfulness",
        f"{rep.mean_cross_checked_faithfulness:.1%}",
        f"LLM: {rep.mean_llm_faithfulness:.1%} | NLI: {rep.mean_nli_faithfulness:.1%}",
    )
    c4.metric(
        "Judge-NLI Disagreements",
        f"{rep.total_judge_disagreements}/{rep.total_claims_evaluated}",
        f"{rep.disagreement_rate:.1%} divergence rate",
    )

    st.subheader(f"2. Representative Failure Triage ({selected_repo})")
    if not rep.representative_failures:
        st.success("All benchmark queries passed retrieval and dual-judge faithfulness checks!")
    else:
        for bucket in rep.representative_failures:
            with st.expander(
                f"🚨 [{bucket.category.value}] {bucket.title} — {bucket.count} queries ({bucket.percentage:.1f}%)",
                expanded=True,
            ):
                st.markdown(f"**Why this happens:** {bucket.description}")
                st.info(f"**Recommended Engineering Fix:** {bucket.remediation}")
                for idx, ex in enumerate(bucket.examples, start=1):
                    st.markdown(f"#### Example #{idx} (`{ex.query_id}`): *{ex.question}*")
                    st.markdown(
                        f"- **Retrieved Chunks:** `{ex.retrieved_chunk_ids}` "
                        f"(Golden: `{ex.golden_chunk_ids}`, First Hit Rank: `{ex.first_relevant_rank or 'MISSED'}`)\n"
                        f"- **Generated Answer:** {ex.generated_answer}\n"
                        f"- **Ground Truth:** {ex.ground_truth_answer}\n"
                        f"- **Scores:** LLM-Judge `{ex.llm_judge_score:.2f}` | "
                        f"NLI-Entailment `{ex.nli_entailment_score:.2f}` | "
                        f"**Cross-Checked `{ex.cross_checked_score:.2f}`**\n"
                        f"- **Root Cause Diagnosis:** :red[{ex.failure_explanation}]"
                    )
                    for fc in ex.flagged_claims:
                        badge = f"⚡ **DISAGREEMENT ({fc.disagreement_type})**" if not fc.agreement else "✓ Agree"
                        st.caption(
                            f"↳ Claim: \"{fc.claim_text}\" → LLM: `{fc.llm_verdict.value}` | "
                            f"NLI: `{fc.nli_label.value}` (E={fc.nli_probabilities.entailment:.2f}, "
                            f"C={fc.nli_probabilities.contradiction:.2f}) | {badge}"
                        )

with tab_sandbox:
    st.subheader("⚡ Interactive Dual-Engine Faithfulness Cross-Checker")
    st.write(
        "Paste any retrieved context chunk and a RAG-generated answer to see how atomic claim decomposition "
        "and **LLM-as-a-Judge × NLI Entailment** catch subtle entity/tier swaps and contradictions."
    )
    default_ctx = (
        "Standard tier subscriptions include a 30-day full refund window from the initial purchase date. "
        "Pro tier subscriptions are eligible for a prorated refund within 14 days. "
        "Enterprise tier annual contracts are strictly non-refundable once provisioned."
    )
    default_ans = (
        "Enterprise tier annual contracts include a 30-day full refund window from the initial purchase date. "
        "Pro tier subscriptions are eligible for a prorated refund within 14 days."
    )
    ctx_input = st.text_area("Retrieved Context (Premise)", value=default_ctx, height=110)
    ans_input = st.text_area("Generated RAG Answer (Hypothesis)", value=default_ans, height=90)

    if st.button("Run Dual-Judge Evaluation", type="primary"):
        engine = DualFaithfulnessEngine()
        res = engine.evaluate(
            generated_answer=ans_input,
            retrieved_chunks=[RetrievedChunk(chunk_id="sandbox_chunk_1", text=ctx_input, score=1.0, rank=1)],
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LLM Judge Score", f"{res.llm_judge_score:.2f}")
        m2.metric("NLI Entailment Score", f"{res.nli_entailment_score:.2f}")
        m3.metric("Cross-Checked Score", f"{res.cross_checked_score:.2f}")
        m4.metric("Judge Disagreements", f"{res.disagreement_count}/{len(res.claims)}")

        for i, cl in enumerate(res.claims, start=1):
            if not cl.agreement:
                st.error(
                    f"**Claim #{i}:** \"{cl.claim_text}\"\n\n"
                    f"- **LLM Judge:** `{cl.llm_verdict.value}` (conf={cl.llm_confidence:.2f})\n"
                    f"- **NLI Cross-Encoder:** `{cl.nli_label.value}` "
                    f"(Entailment={cl.nli_probabilities.entailment:.2f}, "
                    f"Neutral={cl.nli_probabilities.neutral:.2f}, "
                    f"Contradiction={cl.nli_probabilities.contradiction:.2f})\n"
                    f"- **Disagreement Type:** `{cl.disagreement_type}` — {cl.llm_reasoning}"
                )
            else:
                st.success(
                    f"**Claim #{i}:** \"{cl.claim_text}\"\n\n"
                    f"- **LLM Judge:** `{cl.llm_verdict.value}` | **NLI Model:** `{cl.nli_label.value}` "
                    f"(Entailment={cl.nli_probabilities.entailment:.2f}) → **Calibrated Score: {cl.calibrated_score:.2f}**"
                )

with tab_upload:
    st.subheader("📁 Evaluate Your Own RAG Execution Traces (JSONL)")
    st.write(
        "Upload a `.jsonl` trace file where each line has `query_id`, `question`, `generated_answer`, "
        "and `retrieved_chunks`."
    )
    uploaded = st.file_uploader("Upload RAG traces (.jsonl)", type=["jsonl", "json"])
    if uploaded is not None:
        raw_lines = uploaded.getvalue().decode("utf-8").strip().splitlines()
        evaluator = RAGEvaluator(k=5)
        eval_results = []
        q_map = {q.query_id: q for q in BENCHMARK_QUESTIONS}
        for idx, line in enumerate(raw_lines, start=1):
            rec = json.loads(line)
            qid = rec.get("query_id", f"q_{idx}")
            trace = RAGTrace.model_validate(rec)
            q_item = q_map.get(
                qid,
                QuestionItem(
                    query_id=qid,
                    question=trace.question,
                    ground_truth_answer=rec.get("ground_truth_answer", ""),
                    golden_chunk_ids=rec.get("golden_chunk_ids", []),
                ),
            )
            eval_results.append(evaluator.evaluate_single(q_item, trace))
        custom_report = evaluator.summarize_results(eval_results, pipeline_name=uploaded.name)
        st.success(
            f"Evaluated {custom_report.total_queries} traces — "
            f"Pass Rate: {custom_report.pass_rate:.1%} | "
            f"Cross-Checked Faithfulness: {custom_report.mean_cross_checked_faithfulness:.1%}"
        )
