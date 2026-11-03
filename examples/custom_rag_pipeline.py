"""Minimal example showing how to connect ANY custom Python RAG pipeline to `rag-harness`.

Run from terminal:
    rag-harness run --pipeline examples/custom_rag_pipeline.py:run_rag
"""

from __future__ import annotations

from typing import Any, Dict

from rag_harness.demos.public_repos import HybridBM25RerankRAGPipeline

PIPELINE_NAME = "custom-enterprise-rag"
_engine = HybridBM25RerankRAGPipeline()


def run_rag(question: str) -> Dict[str, Any]:
    """Universal adapter contract: accepts `question: str` and returns answer + retrieved chunks."""
    trace = _engine.query(question)
    return {
        "answer": trace.generated_answer,
        "retrieved_chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "score": c.score,
            }
            for c in trace.retrieved_chunks
        ],
        "latency_ms": trace.latency_ms,
    }
