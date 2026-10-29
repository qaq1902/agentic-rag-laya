# TraceRAG — Agentic RAG × Laya

Exploring JEV-style lightweight decisions with Laya in Agentic RAG.

A portfolio research extension of [Slahnia/agentic-rag](https://github.com/Slahnia/agentic-rag), based on [`f248ce6`](https://github.com/Slahnia/agentic-rag/tree/f248ce65d6bc11571c6ef45ffb70a86674ff5a05). The upstream project supplies the LangGraph workflow, retrieval, SQL, web search, generation, sample data and UI. This extension tests whether a small local decision model can replace selected LLM judgments while preserving that workflow.

**Recommendation:** original **LLM Router → Laya Noul Document Grader → original LLM Generator and Generation Grader**. Router Choice remains an unsuccessful experiment. Both configurable backends still default to `llm`; the recommendation is an explicit demo configuration.

## Contributions and preserved behavior

| Component | Upstream | This extension |
|---|---|---|
| Router | LLM structured output: vectorstore / web_search / sql | Optional Choice with the same RouteQuery; original prompt preserved; initial decision logged independently of final State |
| Document Grader | Permissive LLM relevance judgment, yes / no | Optional Noul P(relevant) >= 0.5, **uncalibrated**; original LLM grader preserved |
| Laya inputs/errors | Not present | Reject truncation and invalid probabilities; propagate inference errors explicitly |
| Graph and downstream processing | Nodes, State, conditional edges, filtering, rewriting, generation and recovery | Unchanged; one graph selects the configured chain |

The upstream vectorstore path remains:

```text
LLM route → retrieve → document grader → generate → LLM grounding condition → END
                          ↓ no relevant documents
                     rewrite (bounded) → retrieve → web fallback when exhausted
```

The Router also selects SQL or direct web search. Grounding is the upstream conditional function, not a new node; its skip/retry behavior is preserved. No fine-tuning, confidence fallback or additional retry budget was introduced. See [attribution](docs/ATTRIBUTION.md) for the exact contribution boundary.

## Setup and demo

Clone the published [qaq1902/agentic-rag-laya](https://github.com/qaq1902/agentic-rag-laya) repository to get this extension. Python **3.11** was evaluated (package minimum: 3.10). Install and start [Ollama](https://ollama.com/), then run on macOS/Linux:

```bash
git clone https://github.com/qaq1902/agentic-rag-laya.git
cd agentic-rag-laya
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[laya,dev]'
cp .env.example .env
ollama pull qwen2.5:3b
```

Review the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE): it limits commercial use. External model weights are acquired separately and are not covered by this repository's MIT license. Initial model/embedding downloads require network access and substantial disk space; web fallback also requires the network.

Edit these values in `.env` for the recommended Apple Silicon demo:

```dotenv
ROUTER_BACKEND=llm
DOCUMENT_GRADER_BACKEND=laya
DOCUMENT_GRADER_THRESHOLD=0.5
LAYA_MODEL=convaiinnovations/laya
LAYA_DEVICE=mps
LLM_MODEL=qwen2.5:3b
SQL_MODEL=qwen2.5:3b
LLM_TEMPERATURE=0
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_DIR=outputs/demo/chroma
SQLITE_PATH=outputs/demo/sample.db
```

Keep the sample source descriptions and `DOCUMENTS_DIR=data/documents` from `.env.example`. Restart after changing model/backend/device settings because loading and chains are cached. Set `DOCUMENT_GRADER_BACKEND=llm` for the original baseline. CPU is the default device; the reported evaluation used MPS/Metal. The model identifier above follows its current revision; exact historical checkpoint acquisition is documented in the [evaluation protocol](docs/evaluation/README.md).

```bash
agentic-rag-ingest
agentic-rag -v "What are the three stages of a classic RAG pipeline?"
```

Ingestion resets the selected Chroma collection; use the isolated demo path above. Optional SQL preparation writes the configured generated database; use a new demo database because the upstream seed script replaces its sample tables:

```bash
python scripts/create_sample_db.py
agentic-rag-ingest-csv
```

These are user setup instructions, not commands executed during release preparation. Streamlit is optional (`pip install -e '.[ui,laya]'`, then `streamlit run app.py`). Fresh installation and UI execution were not revalidated for this draft. The [recorded environment](docs/evaluation/environment.txt) has exact versions; dependency manifests have broader ranges.

## Verified C0–C2 observations

Real serial inference used Laya SDK **0.3.13** on MPS and **qwen2.5:3b**, temperature 0, through Ollama on Metal. C0 used source-extracted adapters with real SDK inference while native imports were incomplete; C1/C2 verified native components and Graph execution.

| Evaluation | Recorded result | Interpretation |
|---|---|---|
| C0 Router, 15 labeled single-source questions | Laya reference criteria A **6/15**; explicit criteria C **5/15**, all C answers SQL | Keep LLM Router; no measured LLM-vs-Choice accuracy comparison on these questions |
| C0 encoding | Full-description criteria B SQL option **51→48 tokens**; concise C passed all guards without truncation | Token completeness did not establish useful routing |
| C1, 30 frozen question-document pairs | LLM precision/recall/F1 **1.0 / 0.65 / 0.7879**; Noul **1.0 / 0.75 / 0.8571**; false negatives **7 / 5** | Selected English set, one annotator, permissive relevance labels; no general accuracy-gain claim |
| C1 warm native-grader median | LLM **222.45 ms**; Noul **82.97 ms** | Different sequential runs/cache conditions; no generalizable speedup claim |
| C2, 3 scenarios × 2 backends | Six native workflows completed; offline functional audit **6/6 PASS**; raw harness **6/6 FAIL** from a post-completion observer assertion | Functional integration passed; observability incomplete |

C2 end-to-end observations, one run per condition, excluding model/index initialization:

| Scenario | LLM grader | Noul grader | Verified behavior |
|---|---:|---:|---|
| RAG pipeline stages | 5.650 s | 4.859 s | Relevant evidence retained; supported answer |
| RAGAS faithfulness | 3.678 s | 2.715 s | Relevant evidence retained; supported answer |
| Irrelevant-corpus recovery | 26.316 s | 10.848 s | Reject all chunks, rewrite twice, live web fallback, answer |

Totals: **28 LLM calls** with the LLM grader versus **11 LLM calls + 17 Noul forwards** with Noul. Low-level embedding forward counts remain **unverified**: a hook missed SentenceTransformer's direct `.forward` dispatch. Curated evidence preserves raw failure statuses separately from the offline audit. Web snippets differed, and recovery answers did not clearly disclose web provenance. Timings include different retained contexts and network latency; they do not establish stable end-to-end speedups or production answer quality.

[Evaluation evidence](docs/evaluation/README.md) includes exact metrics, frozen inputs, probabilities, trajectories and provenance hashes. The upstream README's RAGAS scores are upstream claims, not scores reproduced by this extension; no new RAGAS scores are claimed here.

## Layout and readiness

```text
src/agentic_rag/       upstream workflow + configurable decision backends
tests/                upstream regressions + mocked backend-contract tests
data/                 unchanged upstream sample documents and tables
evaluation/           original upstream RAGAS runner and dataset
docs/evaluation/      curated evidence and standard-library offline verifier
docs/ATTRIBUTION.md    upstream credit and external model-license boundaries
docs/PUBLICATION.md    release file list and outstanding blockers
outputs/              ignored local model stores, results, caches and indexes
```

Verify the published evidence without installing the application or calling models:

```bash
python3 docs/evaluation/verify.py
```

Mock/regression tests (`pytest tests/`) are separate from live evaluation. They are not proof of real model behavior; no application test suite was rerun for this documentation change. Remaining work includes repairing external embedding observation, portable live evaluation runners, explicit web-source disclosure and broader independent evaluation. See [publication readiness](docs/PUBLICATION.md).

## License

Upstream code: **MIT, Copyright (c) 2026 Andrea**; [LICENSE](LICENSE) is preserved unchanged. The package/import names remain `agentic-rag` / `agentic_rag`. TraceRAG contributions are the configurable Laya integration, contract/error tests, bounded evaluation and evidence-backed component selection. External models and dependencies retain their own licenses; see [ATTRIBUTION](docs/ATTRIBUTION.md). No model weights, third-party web snippets, caches or private runtime logs are included in the public evidence package.
