# Attribution and license boundaries

## Upstream software and data

TraceRAG extends [Slahnia/agentic-rag](https://github.com/Slahnia/agentic-rag), baseline [f248ce65d6bc11571c6ef45ffb70a86674ff5a05](https://github.com/Slahnia/agentic-rag/tree/f248ce65d6bc11571c6ef45ffb70a86674ff5a05). Upstream is MIT, **Copyright (c) 2026 Andrea**. The root [LICENSE](../LICENSE) is retained byte-for-byte; keep that notice with copies or substantial portions. The original README remains available at the linked baseline.

Upstream supplied the graph/State/nodes, LLM router and grading prompts, retrieval/ingestion, SQL guard and recovery, web tool, generation, CLI/UI, RAGAS harness and sample Markdown/CSV/seed data. Their presence is not a claim that this extension authored them. Sample source files remain unchanged. The frozen document-grader pairs contain attributed excerpts of the MIT upstream sample Markdown, with file/line references and content hashes.

## TraceRAG additions

- `0a85a6d`: configurable LLM/Laya Choice routing, initial-route logging outside State, encoding guards and focused mock tests.
- `826a360`: configurable Laya Noul document grading, uncalibrated 0.5 threshold, explicit error propagation and focused tests; both LLM baselines retained.
- Portfolio documentation and curated, bounded C0–C2 evaluation evidence prepared after those implementation commits.

The graph, nodes, State, existing source data, downstream filters and recovery budgets were not redesigned. No model was trained or fine-tuned. No endorsement by the upstream author or model vendors is implied. The distribution/import names remain `agentic-rag` / `agentic_rag`; the inherited package-author metadata is not reassigned.

## Models and external dependencies

| Resource | Verified license reference | Publication treatment |
|---|---|---|
| Laya SDK 0.3.13 | Installed package metadata declares Apache-2.0; [SDK repository](https://github.com/NandhaKishorM/laya) | Dependency only; no vendored SDK source |
| convaiinnovations/laya | [Publisher model card](https://huggingface.co/convaiinnovations/laya) labels Apache-2.0 | Checkpoint identity recorded; no weights redistributed |
| qwen2.5:3b | [Ollama license layer](https://ollama.com/library/qwen2.5:3b/blobs/b5c0e5cf74cf) and [publisher license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) | Qwen Research License, not MIT/Apache; weights obtained separately |
| paraphrase-multilingual-MiniLM-L12-v2 | [Publisher model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) labels Apache-2.0 | Dependency/model reference only; no weights redistributed |

License references checked 2026-10-29. Public model cards can change; the pinned historical artifact identities are in [conditions.json](evaluation/conditions.json). The Laya model-card check is a current publisher declaration, not an independently recovered license file for every cached historical artifact.

Qwen's license grants research/evaluation use and requires a separate license for commercial use. Redistribution of Qwen materials carries agreement/notice conditions. This repository ships integration code and selected evaluation observations, not Qwen materials or weights; its MIT license does not grant model rights. Commercial service/deployment needs separate review before publication as a product. No blanket “all models are open source/commercially unrestricted” claim is made.

Other installed packages retain their own terms; the environment snapshot is a version record, not a complete third-party license audit. No executable environment is redistributed. A future binary/container/weight distribution needs its own dependency/license review.

## Evidence publication

Numeric observations, source-derived labeled pairs, criteria and short decision outputs are curated from the preserved experiments. Private service/proxy logs, workstation paths, full runtime dumps, third-party Web snippets and recovery answer text are omitted. Web recovery behavior/counts are recorded without redistributing retrieved pages. Public summaries distinguish original harness failures from the later offline functional audit.
