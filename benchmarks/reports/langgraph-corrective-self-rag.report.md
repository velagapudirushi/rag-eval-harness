# RAG Evaluation Report: `langgraph-corrective-self-rag`

- **Dataset:** `cloud_rag_benchmark_v1` (15 queries, cutoff `K=5`)
- **Overall Pass Rate:** **100.0%** (15/15)
- **Cross-Checked Faithfulness:** **97.8%** (LLM-Judge: `100.0%` vs. NLI-Entailment: `95.0%`)
- **Judge-NLI Disagreements Flagged:** `0` / `25` atomic claims (`0.0%`)
- **Mean Latency:** `142.8 ms`

## 1. Retrieval & Faithfulness Summary

| Category | Metric | Score | Notes |
| :--- | :--- | :---: | :--- |
| **Retrieval** | `Recall@5` | **100.0%** | Fraction of golden chunks retrieved in top-5 |
| **Retrieval** | `Precision@5` | `33.3%` | Signal-to-noise ratio in top-5 window |
| **Retrieval** | `MRR` | **1.000** | Reciprocal rank of first golden chunk |
| **Retrieval** | `NDCG@5` | **1.000** | Rank-aware retrieval quality |
| **Retrieval** | `Context Fact Coverage` | `100.0%` | Reference key facts present in retrieved text |
| **Faithfulness** | `Method A: LLM-as-a-Judge` | `100.0%` | Holistic passage-level judge |
| **Faithfulness** | `Method B: NLI Entailment` | `95.0%` | Sentence-aligned premise-hypothesis cross-encoder |
| **Faithfulness** | **`Cross-Checked Score`** | **97.8%** | Calibrated ensemble penalizing NLI contradictions |

## 2. Representative Failures

All queries passed retrieval and faithfulness checks.