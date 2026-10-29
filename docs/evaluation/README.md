# C0–C2 evaluation record

This is a small public derivative of existing experiments from 2026-10-28–29 (Singapore time), code commit `826a3603acaf7d0a92f13bf9de42f6d6f634af6f`. Original experiments remain unchanged in ignored local `outputs/` and a separately verified backup. No model inference, tuning or experiment was performed to prepare this package.

## Evidence map

| Files | Contents |
|---|---|
| [conditions.json](conditions.json), [environment.txt](environment.txt) | Fixed model/checkpoint/device/SDK identities, execution conditions and full observed package versions |
| [c0/router_plan.json](c0/router_plan.json) | Byte-identical frozen 15-question plan, labels, criteria C and rationales |
| [c0/router_results.json](c0/router_results.json) | Exact probabilities/margins/latencies, all 60 measured calls, criteria/source descriptions and integrity summaries |
| [c0/smoke_and_diagnostic.json](c0/smoke_and_diagnostic.json) | Initial real-MPS smoke and A/B encoding diagnosis, including rejected B criteria |
| [c1/frozen_pairs.json](c1/frozen_pairs.json) | Byte-identical frozen 30 source-derived question/document pairs with labels and hashes |
| [c1/predictions.json](c1/predictions.json), [c1/results.csv](c1/results.csv), [c1/metrics.json](c1/metrics.json) | Observed raw short SDK/LLM outputs, warmup flags, decisions, exact timings and confusion matrices |
| [c2/plan.json](c2/plan.json) | Byte-identical fixed graph scenarios/protocol; relative outputs paths are historical identifiers, not included indexes |
| [c2/runs.json](c2/runs.json), [c2/summary.csv](c2/summary.csv), [c2/index_integrity.json](c2/index_integrity.json) | Six run projections, separate raw/audit status, decisions, ordered document hashes, traces, counts and source/index integrity |
| [manifest.json](manifest.json), [SHA256SUMS](SHA256SUMS), [verify.py](verify.py) | Source-artifact hashes, explicit transformations, public-file hashes and offline metric/integrity verification |

Frozen files and environment.txt are byte-identical copies. Other files are documented selected-field projections/CSV joins; they are not complete raw runtime records. Absolute paths, endpoint/proxy details, full tracebacks, internal UUID metadata, Web snippets and final graph answer text are deliberately omitted from the public projections. Original FAIL statuses and metric values are preserved. Source hashes establish traceability if originals are available; they cannot independently prove a historical inference or reconstruct omitted files.

## Conclusions and boundaries

**Router Choice stays experimental.** A (reference-style Laya criteria) scored vectorstore 1/5, SQL 0/5, Web 5/5 = 6/15. C (concise explicit criteria) scored 0/5, 5/5, 0/5 = 5/15, returning SQL for every input. A is not the original LLM Router. No LLM-router accuracy benchmark was run on these 15 questions. Criteria/labels were frozen before A/C predictions; two earlier diagnostic questions were already known. One annotator, no second human adjudication.

C0 used real SDK/MPS via source-extracted production adapter logic because native imports initially lacked langchain_ollama; C1 later completed native imports. A/B used two questions, one excluded warmup plus three measured repetitions per question/variant. Full-description B got SQL question R2 right by only 0.0067 margin, but its SQL option truncated 51→48 tokens and the production guard rejected it; direct SDK execution was diagnostic only. A/C retained complete source descriptions in state, unchanged max_len512/head_max_len192 and 48-token option cap. A options 12/10/14 tokens, C32/31/20; all 90 actual forward tensors matched audited encoding. C had no truncation and still failed discrimination. Thirty warmups + sixty measurements are fifteen unique questions, not ninety independent samples. A/C warm SDK median 83.9635624724906 / 94.55247950972989 ms excludes validation, normalization and I/O.

**Native Document Grader A/B completed.** Same 30 frozen English question/document payloads: 10 clear relevant, 10 clear irrelevant, 10 related-but-insufficient positives under the original permissive relevance rule. Labels were fixed independently of predictions by one analyst. Each backend had one excluded warmup then 30 serial measurements. The LLM baseline was completed later against preserved Noul results after resolving the configured-model availability issue; runs were not contemporaneous/interleaved.

| Metric | LLM | Noul at uncalibrated 0.5 |
|---|---:|---:|
| Matrix [[TN,FP],[FN,TP]] | [[10,0],[7,13]] | [[10,0],[5,15]] |
| Precision | 1.0 | 1.0 |
| Recall | 0.65 | 0.75 |
| F1 | 0.7878787878787878 | 0.8571428571428571 |
| Accuracy | 23/30 | 25/30 |
| Warm median ms | 222.44631248759106 | 82.96633348800242 |
| Warm p95 ms, nearest rank | 663.3508749655448 | 107.86587500479072 |

No inference failures and no LLM parse fallbacks in the completed 30-pair comparison. LLM false negatives: R05,R06,B01,B03,B04,B05,B09. Noul: R02,B01,B02,B09,B10. R02 is clearly relevant, P=0.4896, rejected at fixed0.5. Both graders can discard related evidence. Paired outcomes: both correct20, LLM-only3, Noul-only5, both wrong2. The easy out-of-domain negatives, repeated documents, small sample and single annotator limit accuracy claims. Noul's two net additional correct pairs do not establish statistical/model superiority.

Warm C1 timing includes native invoke formatting/parser or input guard and synchronization, excluding artifact writes/observation copies. LLM first invoke was 1904.3295829906128 ms including 1276.724375 ms server load. Noul's preserved first isolated load was 70010.45325002633 ms, then first invoke351.555792032741 ms, total70362.00904205907 ms. Different load boundaries, caches and process conditions prevent a general startup or speedup claim. Noul options17/17 tokens, total172–299, no truncation; LLM prompts181–298/output6–32, context4096, no observed overflow.

**Native graph integration passed functionally, with incomplete instrumentation.** S1/S2 both backends used `route → retrieve → grade_documents → generate`; original grounding condition returned useful. LLM kept2/4 and Noul1/4; both retained direct answer evidence. S3 used an intentionally irrelevant index made from an exact copy of upstream product-review CSV text; it is a recovery fixture, not a production KB-quality sample. Both rejected3/3 on three retrieval rounds, rewrote twice (retries0→1→2), then used real Web search and generated an answer. Initial route was vectorstore independently of final State.web_search. Original Router/Generator/Generation Grader remained LLM.

| Run | E2E ms | LLM calls | Noul actual forwards |
|---|---:|---:|---:|
| S1_llm | 5649.922167009208 | 7 | 0 |
| S1_laya | 4858.529917022679 | 3 | 4 |
| S2_laya | 2714.7322079981677 | 3 | 4 |
| S2_llm | 3678.421000018716 | 7 | 0 |
| S3_llm | 26316.098541021347 | 14 | 0 |
| S3_laya | 10847.98762499122 | 5 | 9 |

**All six raw harness records retain FAIL** from a post-completion embedding-count assertion. SentenceTransformer directly dispatched a bound `.forward`, bypassing the optional module pre-hook. Recorded hook count0 is not evidence of zero embedding inference. A separate offline audit of recorded updates/branches/filtering/model calls/final States verified6/6 functional runs; its result is PASS_FUNCTIONAL_WITH_OBSERVABILITY_LIMITATION. This package does not repair that observer or turn raw FAILs into PASS. Actual low-level embedding forward count remains null/UNVERIFIED; logical embedding queries1/1/1/1/3/3 are inferred from retrievals and native code. LLM/Noul counts are directly observed: 28 LLM vs11 LLM+17Noul total.

C2 timing starts at graph.stream and includes retrieval, grading, rewriting, generation, grounding, real Web and observation overhead. Embedding/Noul setup/indexing are excluded; one empty-prompt Ollama preload is separate. One run per backend/scenario, different retained contexts and variable live-Web snippets prohibit stable speedup attribution. Ordered Web URLs matched but snippets differed. Both recovery answers used vague source wording despite public-Web context notes; explicit user-facing provenance remains a shared original-generator issue. SQL and direct-Web initial routes, concurrent calls, production answer quality and full cold starts were not evaluated by these three scenarios.

## Reproduction

Offline, standard library only, no models/application imports or writes:

```bash
python3 docs/evaluation/verify.py
```

This checks published checksums, original frozen-input hashes, source/document hashes against tracked sample files, recomputes C0 accuracy and C1 confusion matrices/latency, and checks curated C2 consistency/status/counts/paired retrieval evidence. It verifies the published record, not past inference truth or the complete original offline graph audit.

For future live replay, first fix the embedding observer in a separate task and prepare portable runners. Historic scripts remain private because they hardcode machine paths and include the known observer defect. Reuse exact commit, labels, checkpoints and conditions; do not silently update models/rubrics or tune0.5 on this set. Acquire model files separately, respecting licenses. One way to request the exact Laya revision after reviewing its terms is:

```python
from huggingface_hub import snapshot_download
snapshot = snapshot_download(
    repo_id="convaiinnovations/laya",
    revision="7b928d828b7b0e022f929d9bd2e44165aa270148",
    local_dir="outputs/models/laya",
)
```

Then set `LAYA_MODEL=outputs/models/laya` before starting a fresh process. Pin embedding revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, preserve800/120 chunking, k4 and max_retries2, clone isolated indexes from exact source copies, run serially, and record initialization separately. Snapshot acquisition and current clean installation were not executed in portfolio preparation. Network is required for first downloads and original Web fallback; live search cannot reproduce historical snippets exactly. If future regression tests use fixed Web fixtures, label them as mocks instead of live evaluation.

Original qwen2.5:3b was initially missing; TUN fake-IP DNS caused a pull rejection. An isolated Ollama with the supported HTTPS proxy setting later fetched that exact model normally; security/TLS/redirect checks remained enabled. Original default service/model inventory was preserved. No local proxy address or service state is assumed in the public demo.

See [publication readiness](../PUBLICATION.md) for remaining engineering and release blockers, and [attribution](../ATTRIBUTION.md) for software/model rights. Upstream RAGAS scores were not rerun and are not claimed by TraceRAG.
