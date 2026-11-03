"""Standalone RAG Evaluation Harness — Production Web Application & Multi-Repo Benchmark Suite."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_harness.adapters import CallablePipelineAdapter, HTTPPipelineAdapter, _coerce_chunks
from rag_harness.demos.corpus import BENCHMARK_QUESTIONS
from rag_harness.demos.public_repos import PUBLIC_REPO_PIPELINES
from rag_harness.evaluator import RAGEvaluator
from rag_harness.metrics.ensemble import DualFaithfulnessEngine
from rag_harness.reporting.markdown import generate_markdown_report
from rag_harness.schemas import HarnessReport, QuestionItem, RAGTrace, RetrievedChunk

st.set_page_config(
    page_title="RAG Eval Harness — Standalone RAG Evaluator & Triage Engine",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Eval Harness — Standalone Evaluation & Failure Triage Engine")
st.caption(
    "Evaluate **any RAG pipeline** (via JSONL/CSV logs, live HTTP API endpoint, or Python callable): "
    "**Retrieval Metrics (`Recall@K`, `MRR`, `NDCG@K`)** • "
    "**Dual-Engine Faithfulness (`LLM-as-a-Judge` × `NLI Entailment`)** • "
    "**Root-Cause Failure Taxonomy**"
)


def render_harness_report(rep: HarnessReport, key_prefix: str = "rep") -> None:
    """Render full KPI metrics, representative failure triage, and trace table for any HarnessReport."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pass Rate", f"{rep.pass_rate:.1%}", f"{rep.passed_queries}/{rep.total_queries} queries")
    c2.metric(f"Recall@{rep.k} / MRR", f"{rep.mean_recall_at_k:.1%}", f"MRR: {rep.mean_mrr:.3f} | NDCG: {rep.mean_ndcg_at_k:.3f}")
    c3.metric(
        "Cross-Checked Faithfulness",
        f"{rep.mean_cross_checked_faithfulness:.1%}",
        f"LLM: {rep.mean_llm_faithfulness:.1%} | NLI: {rep.mean_nli_faithfulness:.1%}",
    )
    c4.metric(
        "Judge-NLI Disagreements",
        f"{rep.total_judge_disagreements}/{rep.total_claims_evaluated}",
        f"{rep.disagreement_rate:.1%} claim divergence",
    )

    st.subheader(f"Representative Failure Triage Report (`{rep.pipeline_name}`)")
    if not rep.representative_failures:
        st.success("All evaluated queries passed retrieval and dual-judge faithfulness checks!")
    else:
        for bucket in rep.representative_failures:
            with st.expander(
                f"🚨 [{bucket.category.value}] {bucket.title} — {bucket.count} queries ({bucket.percentage:.1f}%)",
                expanded=True,
            ):
                st.markdown(f"**Root Cause Mechanism:** {bucket.description}")
                st.info(f"**Recommended Engineering Fix:** {bucket.remediation}")
                for idx, ex in enumerate(bucket.examples, start=1):
                    st.markdown(f"#### Representative Failure #{idx} (`{ex.query_id}`): *{ex.question}*")
                    st.markdown(
                        f"- **Retrieved Chunks:** `{ex.retrieved_chunk_ids}` "
                        f"(Golden: `{ex.golden_chunk_ids}`, First Hit Rank: `{ex.first_relevant_rank or 'MISSED'}`)\n"
                        f"- **Generated Answer:** {ex.generated_answer}\n"
                        f"- **Ground Truth:** {ex.ground_truth_answer or 'N/A'}\n"
                        f"- **Faithfulness Scores:** LLM-Judge `{ex.llm_judge_score:.2f}` | "
                        f"NLI-Entailment `{ex.nli_entailment_score:.2f}` | "
                        f"**Cross-Checked `{ex.cross_checked_score:.2f}`**\n"
                        f"- **Diagnosis:** :red[{ex.failure_explanation}]"
                    )
                    for fc in ex.flagged_claims:
                        badge = f"⚡ **DISAGREEMENT ({fc.disagreement_type})**" if not fc.agreement else "✓ Agree"
                        st.caption(
                            f"↳ Claim: \"{fc.claim_text}\" → LLM: `{fc.llm_verdict.value}` | "
                            f"NLI: `{fc.nli_label.value}` (E={fc.nli_probabilities.entailment:.2f}, "
                            f"C={fc.nli_probabilities.contradiction:.2f}) | {badge}"
                        )

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "⬇️ Download Full Markdown Report (.md)",
        data=generate_markdown_report(rep),
        file_name=f"{rep.pipeline_name}_eval_report.md",
        mime="text/markdown",
        key=f"{key_prefix}_dl_md",
    )
    col_dl2.download_button(
        "⬇️ Download Full JSON Report (.json)",
        data=rep.model_dump_json(indent=2),
        file_name=f"{rep.pipeline_name}_eval_report.json",
        mime="application/json",
        key=f"{key_prefix}_dl_json",
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


tab_custom, tab_http, tab_single, tab_bench = st.tabs(
    [
        "📁 Evaluate Any RAG Pipeline (Upload JSONL / JSON / CSV)",
        "🌐 Evaluate a Live RAG HTTP Endpoint",
        "⚡ Single-Trace Dual-Judge Inspector",
        "📊 3-Repo Public RAG Benchmark",
    ]
)

with tab_custom:
    st.subheader("📁 Evaluate Any RAG Pipeline from Execution Logs (`.jsonl`, `.json`, or `.csv`)")
    st.markdown(
        "Export outputs from **any RAG pipeline in any language** (LangChain, LlamaIndex, FastAPI, Node.js, Go) "
        "with fields `question`, `generated_answer` (or `answer`), and `retrieved_chunks` (or `contexts`), "
        "plus optional `ground_truth_answer` and `golden_chunk_ids`."
    )

    sample_jsonl = "\n".join(
        json.dumps(
            {
                "query_id": "custom_q1",
                "question": "What is the refund window for Enterprise tier annual contracts?",
                "generated_answer": "Enterprise tier annual contracts include a 30-day full refund window.",
                "ground_truth_answer": "Enterprise tier annual contracts are strictly non-refundable once provisioned.",
                "golden_chunk_ids": ["doc_01"],
                "retrieved_chunks": [
                    {
                        "chunk_id": "doc_01",
                        "text": "Standard tier subscriptions include a 30-day full refund window. Enterprise tier annual contracts are strictly non-refundable once provisioned.",
                        "score": 0.92,
                    }
                ],
            }
        )
        for _ in range(1)
    )
    st.download_button(
        "📄 Download Sample Input Template (.jsonl)",
        data=sample_jsonl,
        file_name="sample_rag_traces.jsonl",
        mime="application/jsonl",
    )

    uploaded = st.file_uploader(
        "Upload your RAG output file (.jsonl, .json, or .csv)",
        type=["jsonl", "json", "csv"],
    )
    k_cutoff = st.slider("Retrieval Cutoff K", min_value=1, max_value=10, value=5, key="upload_k")

    if uploaded is not None:
        content = uploaded.getvalue().decode("utf-8").strip()
        records: list[dict] = []
        if uploaded.name.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                chunks_raw = row.get("retrieved_chunks") or row.get("contexts") or "[]"
                try:
                    parsed_chunks = json.loads(chunks_raw) if chunks_raw.strip().startswith("[") else [chunks_raw]
                except Exception:
                    parsed_chunks = [chunks_raw]
                records.append(
                    {
                        "query_id": row.get("query_id") or f"q_{len(records)+1}",
                        "question": row.get("question") or row.get("query") or "",
                        "generated_answer": row.get("generated_answer") or row.get("answer") or "",
                        "ground_truth_answer": row.get("ground_truth_answer") or "",
                        "golden_chunk_ids": json.loads(row["golden_chunk_ids"]) if row.get("golden_chunk_ids", "").startswith("[") else [],
                        "retrieved_chunks": parsed_chunks,
                    }
                )
        elif content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

        evaluator = RAGEvaluator(k=k_cutoff)
        eval_results = []
        q_map = {q.query_id: q for q in BENCHMARK_QUESTIONS}

        for idx, rec in enumerate(records, start=1):
            qid = str(rec.get("query_id") or f"q_{idx}")
            qtext = str(rec.get("question") or rec.get("query") or "")
            ans = str(rec.get("generated_answer") or rec.get("answer") or rec.get("response") or "")
            raw_chunks = rec.get("retrieved_chunks") or rec.get("contexts") or rec.get("chunks") or []
            trace = RAGTrace(
                query_id=qid,
                question=qtext,
                generated_answer=ans,
                retrieved_chunks=_coerce_chunks(raw_chunks),
                latency_ms=float(rec.get("latency_ms", 0.0)),
                pipeline_name=uploaded.name,
            )
            if qid in q_map and not rec.get("ground_truth_answer"):
                q_item = q_map[qid]
            else:
                q_item = QuestionItem(
                    query_id=qid,
                    question=qtext,
                    ground_truth_answer=str(rec.get("ground_truth_answer") or ""),
                    golden_chunk_ids=list(rec.get("golden_chunk_ids") or []),
                    key_facts=list(rec.get("key_facts") or []),
                )
            eval_results.append(evaluator.evaluate_single(q_item, trace))

        custom_report = evaluator.summarize_results(eval_results, pipeline_name=uploaded.name)
        render_harness_report(custom_report, key_prefix="custom_upload")

with tab_http:
    st.subheader("🌐 Evaluate Any Live Deployed RAG API Endpoint")
    st.markdown(
        "Point the evaluator at any running RAG HTTP endpoint (`POST` accepting `{\"question\": \"...\"}` and "
        "returning `{\"answer\": \"...\", \"retrieved_chunks\": [...]}`)."
    )
    endpoint_url = st.text_input("RAG Endpoint URL (POST)", placeholder="https://your-rag-service.com/query")
    questions_json = st.text_area(
        "Benchmark Questions JSON",
        value=json.dumps([q.model_dump() for q in BENCHMARK_QUESTIONS[:3]], indent=2),
        height=200,
    )
    if st.button("Run Live HTTP Endpoint Evaluation", type="primary"):
        if not endpoint_url.strip():
            st.warning("Please enter a valid HTTP endpoint URL.")
        else:
            try:
                parsed_q = [QuestionItem.model_validate(item) for item in json.loads(questions_json)]
                adapter = HTTPPipelineAdapter(endpoint_url.strip(), pipeline_name=endpoint_url.strip())
                evaluator = RAGEvaluator(k=5)
                http_report = evaluator.run(adapter, parsed_q)
                render_harness_report(http_report, key_prefix="http_eval")
            except Exception as exc:
                st.error(f"Failed to reach endpoint `{endpoint_url}`: {exc}")

with tab_single:
    st.subheader("⚡ Single-Trace Dual-Judge Faithfulness Inspector")
    st.write(
        "Paste any retrieved context chunk(s) and a generated RAG answer to run **Atomic Claim Decomposition** + "
        "**LLM-as-a-Judge × NLI Entailment Cross-Checking**."
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

    if st.button("Evaluate Claims (LLM Judge × NLI Entailment)", type="primary"):
        engine = DualFaithfulnessEngine()
        res = engine.evaluate(
            generated_answer=ans_input,
            retrieved_chunks=[RetrievedChunk(chunk_id="chunk_1", text=ctx_input, score=1.0, rank=1)],
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
    st.subheader("Head-to-Head Evaluation Across 3 Public RAG Architectures")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    selected_repo = st.selectbox(
        "Inspect Detailed Failure Report & Traces for:",
        options=list(reports.keys()),
        index=0,
    )
    render_harness_report(reports[selected_repo], key_prefix=f"bench_{selected_repo}")
