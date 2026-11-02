# Portfolio publication record

## Published release

TraceRAG was successfully published to [qaq1902/agentic-rag-laya](https://github.com/qaq1902/agentic-rag-laya). The initial public `main` was verified to match commit [`3ecae0a`](https://github.com/qaq1902/agentic-rag-laya/commit/3ecae0ac6a21126293a1e9b5b92bb9db5e7d63ec), `docs: prepare TraceRAG portfolio release`.

- Local development branch: `exp/laya-router-ab`.
- Publication remote: `portfolio`, pointing to `https://github.com/qaq1902/agentic-rag-laya.git`; upstream `origin` remains `https://github.com/Slahnia/agentic-rag.git`.
- Only the explicit `HEAD:refs/heads/main` refspec was pushed, with automatic tag pushing disabled. No other branch, tag or `refs/codex/*` reference was published; no force push or history rewrite was used.
- The initial release contained the 21 reviewed publication files listed in the historical audit below, alongside the existing implementation history. Original outputs, weights, environments and caches remain excluded from the published branch.

This documentation follow-up updates the README demo setup/status and this publication record using existing local acceptance evidence. These follow-up edits await commit/push confirmation. Experimental results, limitations, attribution, model-license boundaries and original evidence remain unchanged. The existing observability and reproducibility limitations still apply; successful publication does not imply production validation. See [evaluation](evaluation/README.md) and [attribution](ATTRIBUTION.md).

## Local Streamlit demo acceptance — 2026-10-29

The unchanged `app.py` was accepted locally at code commit `8b4907e`, reusing the existing Python 3.11 environment and model caches with Streamlit 1.65.0. The local record is `outputs/demo/demo_20261029_140815/` (`report.md`, `ui_transcript.md`, launch manifest and Laya/Streamlit/Ollama logs). These ignored runtime artifacts remain local and are excluded from the public repository; the following is a summary of that record.

- **PASS — UI and knowledge-base answers:** Streamlit started, its health endpoint returned `200 ok`, and real S1/S2 questions completed `route → retrieve → grade_documents → generate` with nonempty answers, expandable node status and Evidence. S1 identified ingestion, retrieval and generation, citing one `rag_fundamentals.md` chunk. S2 explained whether generated claims are supported by retrieved context, citing one `evaluation_ragas.md` chunk. Original LLM grounding remains the graph's post-generation conditional function.
- **PASS — real Laya Noul:** an independent SDK smoke used Laya 0.3.13 on MPS, passed the production input-completeness guard and returned `p_relevant=0.8633`. This probability belongs only to the independent smoke; it is not a score observed for a UI-retrieved document. The first smoke attempt failed because it selected the embedding-only offline cache; the retry using the existing Laya cache succeeded.
- **Configuration and UI backend evidence:** original LLM Router; Laya Noul Document Grader, threshold 0.5, MPS, checkpoint `7b928d828b7b0e022f929d9bd2e44165aa270148`; original LLM Generation/Grounding using `qwen2.5:3b`, temperature 0. The embedding model/cache revision remained `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Streamlit logs record Laya checkpoint loading; Ollama logs record six successful `/api/chat` requests, three per question. Together with the launch configuration and unchanged code path, this supports the UI's use of Laya document grading. Attribution of those LLM requests to Router, Generation and Grounding follows from the code path; the HTTP logs do not label the roles.
- **Screenshot capture failed; saved visual evidence UNVERIFIED:** the GUI capture attempt failed, and no PNG files were saved in the run's screenshot directory. This limits visual auditability but does not invalidate the recorded core Q&A completion. UI per-document Noul probabilities and actual forward counts were not captured; no verified per-document forward-count claim is made.
- **Scope remains bounded:** fresh installation, SQL/Web/recovery paths in this UI demo, concurrency and production readiness remain unverified. Original C2 raw harness records remain 6/6 FAIL from the historical observer assertion, with the separate offline functional audit unchanged; low-level embedding forwards remain UNVERIFIED. This demo adds no general performance, accuracy or statistical-significance claim. No new tests or inference were run for this documentation follow-up.

## Historical pre-publication audit

The following record is preserved verbatim from the initial release commit. Its statements about an unselected destination, an uncommitted draft and proposed next steps describe the pre-publication state; they are not the current publication status.

<details>
<summary>Pre-publication audit retained from commit 3ecae0a</summary>

# Portfolio release preparation

Status: documentation/evidence draft prepared; **not published or committed**. Runtime implementation remains at `826a360`, branch `exp/laya-router-ab`. Upstream master and the origin remote have not been changed. No experiments, application tests, model tuning, source-data edits or dependency installs were performed during preparation.

## Proposed next commit: exact allowlist

The following are the only proposed files for the next documentation/evidence commit. Implementation commits already exist on this branch and are not repeated here. No files are staged. A suggested subject is `docs: prepare TraceRAG portfolio and curate C0-C2 evidence`.

- `.gitignore`
- `README.md`
- `docs/ATTRIBUTION.md`
- `docs/PUBLICATION.md`
- `docs/evaluation/README.md`
- `docs/evaluation/SHA256SUMS`
- `docs/evaluation/c0/router_plan.json`
- `docs/evaluation/c0/router_results.json`
- `docs/evaluation/c0/smoke_and_diagnostic.json`
- `docs/evaluation/c1/frozen_pairs.json`
- `docs/evaluation/c1/metrics.json`
- `docs/evaluation/c1/predictions.json`
- `docs/evaluation/c1/results.csv`
- `docs/evaluation/c2/index_integrity.json`
- `docs/evaluation/c2/plan.json`
- `docs/evaluation/c2/runs.json`
- `docs/evaluation/c2/summary.csv`
- `docs/evaluation/conditions.json`
- `docs/evaluation/environment.txt`
- `docs/evaluation/manifest.json`
- `docs/evaluation/verify.py`

`README.md` is rewritten for the portfolio; `.gitignore` is extended to exclude local generated assets, environments, weight files and common secret-file names. All docs files are new. Source code, tests, `.env.example`, dependency manifests, LICENSE and original datasets are unchanged. Existing experiment evidence is neither edited nor moved; public files are copies or documented projections.

## Clean structure

Keep `src/`, `tests/`, `data/`, original `evaluation/`, CLI/UI/scripts and package identity in place to preserve attribution and existing paths. Use `docs/evaluation/` for the bounded public record. Use ignored `outputs/` exclusively for live experiment results, isolated indexes, model stores and environments. Machine-specific runners/logs stay private. No mass folder restructuring is needed for this release.

## Publication decisions and blockers

| Item | Finding / required action |
|---|---|
| Internal local snapshots | Local `refs/codex/...` snapshot references include original outputs and model blobs, although the publication branch and master commit chains do not. They were not modified. Never mirror-push all refs or publish an all-ref bundle from this checkout. Publish only a reviewed named branch and inspect a clean clone before release. |
| Destination repository | `origin` still points to Slahnia/agentic-rag. A personal fork/new repository owner and URL have not been specified. Select a publication destination separately; no push to upstream is authorized. README deliberately has no invented clone URL. |
| Upstream attribution | MIT notice Copyright2026 Andrea remains in LICENSE unchanged. README/ATTRIBUTION identify baseline and contribution commits; keep attribution when publishing. |
| Model rights | qwen2.5:3b is Qwen Research License, not Apache/MIT. This is a research/evaluation portfolio, without weights. Commercial deployment/use needs separate licensing review. No blanket commercial permission or all-open-source claim. See ATTRIBUTION links. |
| Dependency distribution | Laya/embedding publisher cards label Apache-2.0, SDK metadata agrees. No vendor source/binary/weights distributed. This is not a full transitive license audit; perform one before packaging environments/containers. |
| Observability | All six original C2 automatic runs retain FAIL from the post-completion embedding-hook assertion. Offline audit supports functional6/6 PASS; low-level embedding count stays UNVERIFIED. This blocks an all-green/production-validation claim, not honest publication of the result. |
| Reproducibility | Standard-library verification is provided. Historic live runners are not portable and are not shipped; model weights/indexes are excluded. Fresh install/UI, full live replay and release packaging were not executed here. The general dependency ranges are not a lockfile. |
| Quality claims | 15 Router questions, 30 grader pairs, one annotator and 3 graph scenarios; no general speedup/accuracy gain, statistical superiority, new RAGAS score, SQL/full-path integration or concurrency claim. |
| Web/privacy | Public projections omit private paths, service/proxy logs, full traceback/messages and Web snippets/answers. Initial/final route, raw failure status and metrics are retained. A local scan checks curated/tracked content for obvious secret patterns and workstation paths; it is not a proof of historical secrecy. |
| Final review | Review the exact allowlist/diff and upstream attribution before a separately authorized commit/push. A professional demo may need the observer/provenance fixes and broader evaluation under a separate scope. |

## Suggested release checks

`python3 docs/evaluation/verify.py` verifies the evidence offline. `git diff --check` checks whitespace. Mock tests remain separate from real C0–C2 inference evidence and were not rerun for this docs-only preparation. Never broadly stage ignored outputs or force-add model files. Before future publication, re-check that the staged names equal the allowlist above and inspect all staged content. No broad staging command is provided here.

Gitignore handles generated/secret filenames; it cannot sanitize private values embedded in arbitrary text or erase prior Git history. The publication branch history was scanned for selected workstation-path, token and private-key patterns; internal snapshot blob contents were not fully scanned. This is not a complete secret audit. Repeat review if other changes are added before publication. Preserve original private evidence and the external durable evidence backup rather than replacing it with the public projections.

</details>
