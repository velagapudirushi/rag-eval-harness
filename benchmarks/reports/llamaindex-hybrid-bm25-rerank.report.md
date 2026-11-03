# RAG Evaluation Report: `llamaindex-hybrid-bm25-rerank`

- **Dataset:** `cloud_rag_benchmark_v1` (15 queries, cutoff `K=5`)
- **Overall Pass Rate:** **80.0%** (12/15)
- **Cross-Checked Faithfulness:** **90.5%** (LLM-Judge: `100.0%` vs. NLI-Entailment: `88.1%`)
- **Judge-NLI Disagreements Flagged:** `3` / `26` atomic claims (`11.5%`)
- **Mean Latency:** `78.2 ms`

## 1. Retrieval & Faithfulness Summary

| Category | Metric | Score | Notes |
| :--- | :--- | :---: | :--- |
| **Retrieval** | `Recall@5` | **100.0%** | Fraction of golden chunks retrieved in top-5 |
| **Retrieval** | `Precision@5` | `20.0%` | Signal-to-noise ratio in top-5 window |
| **Retrieval** | `MRR` | **1.000** | Reciprocal rank of first golden chunk |
| **Retrieval** | `NDCG@5` | **1.000** | Rank-aware retrieval quality |
| **Retrieval** | `Context Fact Coverage` | `100.0%` | Reference key facts present in retrieved text |
| **Faithfulness** | `Method A: LLM-as-a-Judge` | `100.0%` | Holistic passage-level judge |
| **Faithfulness** | `Method B: NLI Entailment` | `88.1%` | Sentence-aligned premise-hypothesis cross-encoder |
| **Faithfulness** | **`Cross-Checked Score`** | **90.5%** | Calibrated ensemble penalizing NLI contradictions |

## 2. Representative Failure Taxonomy

### `JUDGE_NLI_DISAGREEMENT` — Judge-NLI Disagreement (LLM Judge Fooled by Plausible Hallucination) (2 queries / 13.3%)

> **Why this happens:** The LLM-as-a-Judge marked the claim SUPPORTED due to passage-level keyword overlap, but the sentence-aligned NLI Entailment cross-encoder caught a contradiction or scope swap.
>
> **Recommended Fix:** Enforce sentence-level citation grounding in the generation prompt and gate CI on NLI entailment cross-checks rather than single-model LLM judge scores.

#### Representative Failure #1: `q_15` — *Can a user with the Auditor RBAC role view raw document chunks or delete a collection?*
- **Retrieved Chunks:** `['doc_15_rbac_roles', 'doc_09_error_err_idx_809', 'doc_03_cli_2fa_reset', 'doc_16_distractor_billing_faq', 'doc_02_rate_limits']` (Golden: `['doc_15_rbac_roles']`, First Hit Rank: `1`)
- **Generated Answer:** The Auditor role grants read-only access to billing invoices, security logs, and evaluation reports. The Auditor role can execute API queries but cannot delete collections.
- **Ground Truth:** No. The Auditor role has read-only access to billing invoices, security logs, and evaluation reports, and cannot view raw document chunks, execute API queries, or delete collections.
- **Faithfulness Breakdown:** LLM Judge `1.00` | NLI Entailment `0.33` | **Cross-Checked `0.34`**
- **Root Cause Diagnosis:** LLM Judge scored claim as SUPPORTED, but NLI Cross-Encoder flagged CONTRADICTION (p_contra=0.94, p_neutral=0.04) on: "The Auditor role can execute API queries."
- **Flagged Atomic Claims:**
  - `"The Auditor role can execute API queries."` → LLM: `SUPPORTED`, NLI: `CONTRADICTION` (`E=0.01`, `C=0.94`) ⚠️ **[LLM_MISSED_CONTRADICTION]**
  - `"cannot delete collections."` → LLM: `SUPPORTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.89`) ⚠️ **[LLM_MISSED_CONTRADICTION]**

#### Representative Failure #2: `q_11` — *What uptime SLA and service credit does a Standard tier workspace receive if uptime drops below 99.0%?*
- **Retrieved Chunks:** `['doc_11_sla_credits', 'doc_01_refund_policy', 'doc_02_rate_limits', 'doc_16_distractor_billing_faq', 'doc_15_rbac_roles']` (Golden: `['doc_11_sla_credits']`, First Hit Rank: `1`)
- **Generated Answer:** Standard tier workspaces do not carry a financially backed uptime SLA. However, Standard workspaces receive a 10% service credit if monthly uptime falls between 99.90% and 99.99%.
- **Ground Truth:** Standard tier workspaces do not carry a financially backed uptime SLA and receive no service credits. The 99.99% SLA and 10%/30% service credits apply only to Enterprise workspaces.
- **Faithfulness Breakdown:** LLM Judge `1.00` | NLI Entailment `0.49` | **Cross-Checked `0.50`**
- **Root Cause Diagnosis:** LLM Judge scored claim as SUPPORTED, but NLI Cross-Encoder flagged CONTRADICTION (p_contra=0.92, p_neutral=0.05) on: "However, Standard workspaces receive a 10% service credit if monthly uptime falls between 99.90% and 99.99%."
- **Flagged Atomic Claims:**
  - `"However, Standard workspaces receive a 10% service credit if monthly uptime falls between 99.90% and 99.99%."` → LLM: `SUPPORTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.92`) ⚠️ **[LLM_MISSED_CONTRADICTION]**

### `INCOMPLETE_SYNTHESIS` — Incomplete Multi-Chunk Synthesis (Partial Answer) (1 queries / 6.7%)

> **Why this happens:** The query required combining facts across multiple chunks, and the generator answered only part of the question while omitting critical constraints.
>
> **Recommended Fix:** Increase top-K after reranking or use decompose-then-synthesize prompting for multi-hop queries.

#### Representative Failure #1: `q_13` — *What Helm flags and CUDA/driver versions are needed to enable GPU for the self-hosted reranker pod?*
- **Retrieved Chunks:** `['doc_13_helm_gpu_flag', 'doc_06_data_retention', 'doc_18_distractor_sdk_retries', 'doc_02_rate_limits', 'doc_12_backup_rpo_rto']` (Golden: `['doc_13_helm_gpu_flag']`, First Hit Rank: `1`)
- **Generated Answer:** Pass `--set reranker.gpu.enabled=true --set reranker.resources.vramMinGb=8` during `helm upgrade --install` to deploy the reranker pod on GPU nodes.
- **Ground Truth:** Pass `--set reranker.gpu.enabled=true --set reranker.resources.vramMinGb=8` with `helm upgrade --install`. It requires NVIDIA driver 535.104+ and CUDA 12.1.
- **Faithfulness Breakdown:** LLM Judge `1.00` | NLI Entailment `0.98` | **Cross-Checked `0.99`**
- **Root Cause Diagnosis:** Partial synthesis: Recall@5=1.00, Answer Fact Completeness=50%.
