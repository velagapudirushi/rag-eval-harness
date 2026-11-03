# RAG Evaluation Report: `langchain-naive-chroma-rag`

- **Dataset:** `cloud_rag_benchmark_v1` (15 queries, cutoff `K=5`)
- **Overall Pass Rate:** **46.7%** (7/15)
- **Cross-Checked Faithfulness:** **68.9%** (LLM-Judge: `82.4%` vs. NLI-Entailment: `69.0%`)
- **Judge-NLI Disagreements Flagged:** `4` / `31` atomic claims (`12.9%`)
- **Mean Latency:** `34.5 ms`

## 1. Retrieval & Faithfulness Summary

| Category | Metric | Score | Notes |
| :--- | :--- | :---: | :--- |
| **Retrieval** | `Recall@5` | **86.7%** | Fraction of golden chunks retrieved in top-5 |
| **Retrieval** | `Precision@5` | `17.3%` | Signal-to-noise ratio in top-5 window |
| **Retrieval** | `MRR` | **0.817** | Reciprocal rank of first golden chunk |
| **Retrieval** | `NDCG@5` | **0.829** | Rank-aware retrieval quality |
| **Retrieval** | `Context Fact Coverage` | `86.7%` | Reference key facts present in retrieved text |
| **Faithfulness** | `Method A: LLM-as-a-Judge` | `82.4%` | Holistic passage-level judge |
| **Faithfulness** | `Method B: NLI Entailment` | `69.0%` | Sentence-aligned premise-hypothesis cross-encoder |
| **Faithfulness** | **`Cross-Checked Score`** | **68.9%** | Calibrated ensemble penalizing NLI contradictions |

## 2. Representative Failure Taxonomy

### `JUDGE_NLI_DISAGREEMENT` — Judge-NLI Disagreement (LLM Judge Fooled by Plausible Hallucination) (2 queries / 13.3%)

> **Why this happens:** The LLM-as-a-Judge marked the claim SUPPORTED due to passage-level keyword overlap, but the sentence-aligned NLI Entailment cross-encoder caught a contradiction or scope swap.
>
> **Recommended Fix:** Enforce sentence-level citation grounding in the generation prompt and gate CI on NLI entailment cross-checks rather than single-model LLM judge scores.

#### Representative Failure #1: `q_01` — *What is the refund window for Enterprise tier annual contracts?*
- **Retrieved Chunks:** `['doc_01_refund_policy', 'doc_11_sla_credits', 'doc_07_sso_saml', 'doc_06_data_retention', 'doc_04_encryption_kms']` (Golden: `['doc_01_refund_policy']`, First Hit Rank: `1`)
- **Generated Answer:** Enterprise tier annual contracts include a 30-day full refund window from the purchase date. Unused credits can also be reallocated across workspaces.
- **Ground Truth:** Enterprise tier annual contracts are strictly non-refundable once provisioned, though unused credits can be reallocated across workspaces.
- **Faithfulness Breakdown:** LLM Judge `1.00` | NLI Entailment `0.00` | **Cross-Checked `0.01`**
- **Root Cause Diagnosis:** LLM Judge scored claim as SUPPORTED, but NLI Cross-Encoder flagged CONTRADICTION (p_contra=0.92, p_neutral=0.05) on: "Enterprise tier annual contracts include a 30-day full refund window from the purchase date."
- **Flagged Atomic Claims:**
  - `"Enterprise tier annual contracts include a 30-day full refund window from the purchase date."` → LLM: `SUPPORTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.92`) ⚠️ **[LLM_MISSED_CONTRADICTION]**
  - `"Unused credits can also be reallocated across workspaces."` → LLM: `SUPPORTED`, NLI: `CONTRADICTION` (`E=0.02`, `C=0.91`) ⚠️ **[LLM_MISSED_CONTRADICTION]**

#### Representative Failure #2: `q_11` — *What uptime SLA and service credit does a Standard tier workspace receive if uptime drops below 99.0%?*
- **Retrieved Chunks:** `['doc_11_sla_credits', 'doc_01_refund_policy', 'doc_16_distractor_billing_faq', 'doc_09_error_err_idx_809', 'doc_06_data_retention']` (Golden: `['doc_11_sla_credits']`, First Hit Rank: `1`)
- **Generated Answer:** Standard tier workspaces carry a 99.99% monthly uptime SLA and receive a 30% service credit if monthly uptime drops below 99.0%.
- **Ground Truth:** Standard tier workspaces do not carry a financially backed uptime SLA and receive no service credits. The 99.99% SLA and 10%/30% service credits apply only to Enterprise workspaces.
- **Faithfulness Breakdown:** LLM Judge `1.00` | NLI Entailment `0.00` | **Cross-Checked `0.01`**
- **Root Cause Diagnosis:** LLM Judge scored claim as SUPPORTED, but NLI Cross-Encoder flagged CONTRADICTION (p_contra=0.92, p_neutral=0.05) on: "Standard tier workspaces carry a 99.99% monthly uptime SLA and receive a 30% service credit if monthly uptime drops below 99.0%."
- **Flagged Atomic Claims:**
  - `"Standard tier workspaces carry a 99.99% monthly uptime SLA and receive a 30% service credit if monthly uptime drops below 99.0%."` → LLM: `SUPPORTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.92`) ⚠️ **[LLM_MISSED_CONTRADICTION]**

### `RETRIEVAL_MISS` — Retrieval Miss (Golden Evidence Absent from Top-K Chunks) (2 queries / 13.3%)

> **Why this happens:** The retriever failed to surface the ground-truth document chunk in the top-K results, starving the generator of required facts.
>
> **Recommended Fix:** Switch from pure dense embeddings to Hybrid Search (BM25 + Dense with Reciprocal Rank Fusion) and add query expansion for exact error codes, CLI flags, and acronyms.

#### Representative Failure #1: `q_09` — *What causes error code ERR_IDX_809 and how do you fix it?*
- **Retrieved Chunks:** `['doc_08_vector_index_hnsw', 'doc_17_distractor_cli_install', 'doc_18_distractor_sdk_retries', 'doc_13_helm_gpu_flag', 'doc_04_encryption_kms']` (Golden: `['doc_09_error_err_idx_809']`, First Hit Rank: `MISSED`)
- **Generated Answer:** Error code ERR_IDX_809 indicates an HNSW graph memory overflow when ef_construction exceeds 200. Increase client timeout to 60 seconds to resolve it.
- **Ground Truth:** ERR_IDX_809 is caused by ingesting embedding vectors whose dimension does not match the collection schema (e.g., 1536-dim into a 768-dim index). Fix it via `vaultctl collection migrate --reembed` or `--dim 1536`.
- **Faithfulness Breakdown:** LLM Judge `0.15` | NLI Entailment `0.06` | **Cross-Checked `0.03`**
- **Root Cause Diagnosis:** Retriever missed golden chunk(s) ['doc_09_error_err_idx_809'] (Recall@5=0.00), causing generator to hallucinate unsupported facts (Faithfulness=0.03).
- **Flagged Atomic Claims:**
  - `"Error code ERR_IDX_809 indicates an HNSW graph memory overflow when ef_construction exceeds 200."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.06`, `C=0.06`)
  - `"Increase client timeout to 60 seconds to resolve it."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.19`, `C=0.05`)

#### Representative Failure #2: `q_03` — *What exact CLI command and reason code resets 2FA for a locked-out user?*
- **Retrieved Chunks:** `['doc_17_distractor_cli_install', 'doc_07_sso_saml', 'doc_15_rbac_roles', 'doc_06_data_retention', 'doc_18_distractor_sdk_retries']` (Golden: `['doc_03_cli_2fa_reset']`, First Hit Rank: `MISSED`)
- **Generated Answer:** You can reset 2FA for a locked-out user by running `vaultctl login --sso --reset-2fa` in your terminal session.
- **Ground Truth:** Run `vaultctl auth mfa-reset --user <email> --reason-code SEC-402`, which immediately revokes active session tokens and sends a one-time hardware key enrollment link.
- **Faithfulness Breakdown:** LLM Judge `1.00` | NLI Entailment `0.83` | **Cross-Checked `0.92`**
- **Root Cause Diagnosis:** Retriever failed to fetch golden chunk(s) ['doc_03_cli_2fa_reset'] in top-5 (retrieved: ['doc_17_distractor_cli_install', 'doc_07_sso_saml', 'doc_15_rbac_roles', 'doc_06_data_retention', 'doc_18_distractor_sdk_retries']).

### `INTRINSIC_CONTRADICTION` — Intrinsic Contradiction (Misread Numbers, Negations, or Tiers) (1 queries / 6.7%)

> **Why this happens:** The retriever fetched the relevant passage, but the generator misread a numeric figure, inverted a negation polarity, or confused policy rules across adjacent tiers.
>
> **Recommended Fix:** Keep structured tables/policies intact during chunking (semantic/markdown-header chunking) and instruct the generator to verify entity scope before stating numbers.

#### Representative Failure #1: `q_14` — *Is the /v1/graphql endpoint still accepting mutations, and what port does gRPC ingestion use?*
- **Retrieved Chunks:** `['doc_14_graphql_deprecation', 'doc_09_error_err_idx_809', 'doc_05_webhook_retries', 'doc_01_refund_policy', 'doc_02_rate_limits']` (Golden: `['doc_14_graphql_deprecation']`, First Hit Rank: `1`)
- **Generated Answer:** The /v1/graphql endpoint is active for mutations until December 31, and gRPC ingestion streams run on port 8443.
- **Ground Truth:** No, `/v1/graphql` is deprecated and read-only as of Q2 (sunset December 31). Mutations must use `/v2/rest/mutations` or the gRPC ingestion stream on port 9443.
- **Faithfulness Breakdown:** LLM Judge `0.00` | NLI Entailment `0.00` | **Cross-Checked `0.00`**
- **Root Cause Diagnosis:** Generated answer contradicts retrieved evidence (1 contradicted claim(s)): "The /v1/graphql endpoint is active for mutations until December 31, and gRPC ingestion streams run on port 8443."
- **Flagged Atomic Claims:**
  - `"The /v1/graphql endpoint is active for mutations until December 31, and gRPC ingestion streams run on port 8443."` → LLM: `CONTRADICTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.91`)

### `EXTRINSIC_HALLUCINATION` — Extrinsic Hallucination (Unsupported Facts Invented Outside Context) (1 queries / 6.7%)

> **Why this happens:** The generator fabricated specific claims, commands, or parameters that do not exist anywhere in the retrieved context chunks.
>
> **Recommended Fix:** Add strict abstention instructions ('Answer ONLY using provided chunks; if missing, state what cannot be verified') and lower generation temperature to 0.0.

#### Representative Failure #1: `q_08` — *What are the default HNSW index parameters and what happens when ef_search is raised to 128?*
- **Retrieved Chunks:** `['doc_08_vector_index_hnsw', 'doc_06_data_retention', 'doc_09_error_err_idx_809', 'doc_02_rate_limits', 'doc_01_refund_policy']` (Golden: `['doc_08_vector_index_hnsw']`, First Hit Rank: `1`)
- **Generated Answer:** The default HNSW index uses m=16, ef_construction=200, and ef_search=64. Raising ef_search to 128 improves Recall@10 by 4.2% at 18ms extra latency. It also automatically enables int8 scalar quantization on port 6333.
- **Ground Truth:** The default HNSW index uses m=16, ef_construction=200, and ef_search=64. Raising ef_search to 128 increases Recall@10 by about 4.2% with 18ms extra p99 latency.
- **Faithfulness Breakdown:** LLM Judge `0.72` | NLI Entailment `0.64` | **Cross-Checked `0.65`**
- **Root Cause Diagnosis:** Generator fabricated 1 unverifiable claim(s) not grounded in retrieved context: "It also automatically enables int8 scalar quantization on port 6333."
- **Flagged Atomic Claims:**
  - `"It also automatically enables int8 scalar quantization on port 6333."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.06`, `C=0.06`)

### `LOST_IN_THE_MIDDLE` — Context Ignored / Lost-in-the-Middle (Evidence Retrieved but Unused) (1 queries / 6.7%)

> **Why this happens:** The relevant golden chunk was successfully retrieved in the top-K window (often at rank 3–5), but the generator ignored those facts or falsely abstained.
>
> **Recommended Fix:** Add a Cross-Encoder Reranker (e.g., BGE-Reranker or Cohere Rerank) to push the golden chunk to Rank 1–2 and trim low-relevance distractor chunks from the prompt.

#### Representative Failure #1: `q_12` — *What are the RPO, RTO, and Point-in-Time Recovery (PITR) windows for production database clusters?*
- **Retrieved Chunks:** `['doc_10_eu_data_residency', 'doc_11_sla_credits', 'doc_06_data_retention', 'doc_12_backup_rpo_rto', 'doc_04_encryption_kms']` (Golden: `['doc_12_backup_rpo_rto']`, First Hit Rank: `4`)
- **Generated Answer:** I do not have enough information in the provided context to determine the RPO, RTO, and Point-in-Time Recovery windows.
- **Ground Truth:** Production clusters have a 5-minute Recovery Point Objective (RPO), a 30-minute Recovery Time Objective (RTO), and Point-in-Time Recovery (PITR) to any second within the last 14 days.
- **Faithfulness Breakdown:** LLM Judge `1.00` | NLI Entailment `1.00` | **Cross-Checked `1.00`**
- **Root Cause Diagnosis:** Golden evidence was retrieved at rank 4, but generator ignored key facts (completeness=0%, abstained=True).

### `INCOMPLETE_SYNTHESIS` — Incomplete Multi-Chunk Synthesis (Partial Answer) (1 queries / 6.7%)

> **Why this happens:** The query required combining facts across multiple chunks, and the generator answered only part of the question while omitting critical constraints.
>
> **Recommended Fix:** Increase top-K after reranking or use decompose-then-synthesize prompting for multi-hop queries.

#### Representative Failure #1: `q_06` — *Is Zero Data Retention (ZDR) enabled by default on Standard and Pro tiers?*
- **Retrieved Chunks:** `['doc_06_data_retention', 'doc_02_rate_limits', 'doc_08_vector_index_hnsw', 'doc_01_refund_policy', 'doc_10_eu_data_residency']` (Golden: `['doc_06_data_retention']`, First Hit Rank: `1`)
- **Generated Answer:** Zero Data Retention (ZDR) is enabled by default on Standard and Pro tiers for 30 days. Audit security logs are retained for 365 days.
- **Ground Truth:** No, Zero Data Retention (ZDR) is disabled by default and requires signing an Enterprise BAA addendum. Standard and Pro tiers retain inference trace logs for 30 days by default.
- **Faithfulness Breakdown:** LLM Judge `0.50` | NLI Entailment `0.96` | **Cross-Checked `0.78`**
- **Root Cause Diagnosis:** Partial synthesis: Recall@5=1.00, Answer Fact Completeness=33%.
- **Flagged Atomic Claims:**
  - `"Zero Data Retention (ZDR) is enabled by default on Standard and Pro tiers for 30 days."` → LLM: `CONTRADICTED`, NLI: `ENTAILMENT` (`E=0.94`, `C=0.01`) ⚠️ **[LLM_FALSE_NEGATIVE]**
