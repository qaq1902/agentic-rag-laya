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
