# Agentic RAG × Laya

*Lightweight Decision Models for Agentic RAG*

## Overview

Agentic RAG workflows often rely on Large Language Models (LLMs) not only for answer generation, but also for tasks such as data source routing and document relevance grading. However, these decisions typically have a limited output space and do not always require generative inference.

Inspired by JEV-style decision models and discriminative classification, this project explores integrating **Laya**, a lightweight local decision model, into an existing LangGraph-based Agentic RAG system to reduce inference overhead.

**Key Result:** Laya Noul achieved 2.7–4.1× lower warm grading latency ratios compared with the LLM baseline across two evaluations, with a trade-off between speed and relevance recall.

## Architecture

The project explores replacing two LLM-driven decision components:

- **Choice Router:** Uses Laya to select among vector retrieval, web search, and SQL.
- **Noul Document Grader:** Uses Laya to determine whether retrieved documents are relevant to the question.

Both components support configurable LLM and Laya backends without requiring separate LangGraph workflows.

The final architecture adopts a hybrid design:

**LLM Router → Vector Retrieval → Laya Noul Grader → LLM Generation & Grounding**

The original web search, SQL, query rewriting, and recovery paths remain available.

## Conclusion

The experiments showed different outcomes across decision tasks.

Laya Choice did not perform reliably enough for data source routing, while Noul demonstrated better relevance-grading performance and lower latency under the tested conditions.

In the supplementary TREC evaluation, Laya Noul showed more conservative relevance filtering, favoring precision over recall while maintaining its latency advantage.

The resulting hybrid architecture retains LLMs for routing, answer generation, and grounding verification, while delegating document relevance grading to Laya.

The integration was validated through native LangGraph execution and end-to-end Streamlit question answering.

Detailed results are available in the [evaluation records](docs/evaluation/README.md).

## Getting Started

**Requirements:** Python 3.10+, Ollama, and a compatible local environment. Apple Silicon is supported through MPS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[ui,laya]'
cp .env.example .env
ollama pull qwen2.5:3b
```

Configure `.env` for the hybrid architecture on Apple Silicon:

```ini
ROUTER_BACKEND=llm
DOCUMENT_GRADER_BACKEND=laya
DOCUMENT_GRADER_THRESHOLD=0.5
LAYA_DEVICE=mps

CHROMA_DIR=outputs/demo/local/chroma
SQLITE_PATH=outputs/demo/local/sample.db
```

With Ollama running, launch the Streamlit interface:

```bash
streamlit run app.py
```

The application initializes the document index when needed.

## Credits & License

Built on [Slahnia/agentic-rag](https://github.com/Slahnia/agentic-rag), preserving its LangGraph workflow and MIT license.

Laya integration and evaluation are developed in this repository. See [Attribution](docs/ATTRIBUTION.md) for upstream components and external model licenses.

Licensed under the terms in [LICENSE](LICENSE).
