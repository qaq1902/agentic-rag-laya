"""Verify curated evidence offline; no application/model imports, network or writes."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads((HERE / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def verify():
    listed = set()
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        target = (HERE / name).resolve()
        check(HERE in target.parents, "Checksum path outside evidence directory")
        check(name not in listed, "Duplicate checksum entry")
        check(sha(target) == expected, f"Checksum mismatch: {name}")
        listed.add(name)
    actual = {
        p.relative_to(HERE).as_posix()
        for p in HERE.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts
    }
    check(listed == actual, "Checksum list does not cover exactly the public evidence")
    manifest = load("manifest.json")
    for entry in manifest["files"]:
        if entry["operation"] == "byte-identical copy":
            check(sha(HERE / entry["file"]) == entry["source_artifacts"][0]["sha256"], "Original-copy hash differs")

    router = load("c0/router_results.json")
    plan = load("c0/router_plan.json")
    check(sha(HERE / "c0/router_plan.json") == router["frozen_plan_sha256"], "Router frozen hash")
    labels = {p["id"]: p["expected"] for p in plan["cases"]}
    check(len(labels) == 15, "Expected 15 unique router questions")
    for variant in ("A", "C"):
        rows = [r for r in router["per_case_summary"] if r["variant"] == variant]
        check(len(rows) == 15 and {r["id"] for r in rows} == set(labels), "Router coverage")
        for source in ("vectorstore", "sql", "web_search", "all"):
            group = rows if source == "all" else [r for r in rows if labels[r["id"]] == source]
            expected = router["accuracy"][variant][source]
            check(sum(r["decision"] == labels[r["id"]] for r in group) == expected["correct"], "Router accuracy")
            check(len(group) == expected["total"], "Router denominator")
        calls = [c for c in router["measured_calls"] if c["variant"] == variant]
        check(len(calls) == 30 and all(c["phase"] == "measured_warm" for c in calls), "Router repeats")
        coverage = {name: 0 for name in labels}
        for c in calls:
            coverage[c["case"]] += 1
            check(c["expected"] == labels[c["case"]], "Router label consistency")
            check(c["decision"] == max(c["probabilities"], key=c["probabilities"].get), "Choice is not argmax")
            check(all(0 <= p <= 1 for p in c["probabilities"].values()), "Invalid choice probability")
        check(set(coverage.values()) == {2}, "Expected two measured repeats per question")
        times = [c["warm_latency_ms"] for c in calls]
        check(close(statistics.median(times), router["latency"][variant]["median_ms"]), "Router median")
    check(all(r["decision"] == "sql" for r in router["per_case_summary"] if r["variant"] == "C"), "C constant SQL finding")

    frozen = load("c1/frozen_pairs.json")
    metrics = load("c1/metrics.json")
    predictions = load("c1/predictions.json")
    check(sha(HERE / "c1/frozen_pairs.json") == metrics["frozen_pairs_sha256"], "C1 frozen hash")
    pairs = {p["id"]: p for p in frozen["pairs"]}
    check(len(pairs) == 30, "Expected 30 unique grader pairs")
    for p in pairs.values():
        check(sha(ROOT / p["source"]) == p["source_sha256"], "Tracked source hash")
        check(hashlib.sha256(p["document"].encode()).hexdigest() == p["document_sha256"], "Document hash")
    measured = {}
    for backend in ("llm", "laya"):
        calls = predictions[backend]
        check(len(calls) == 31 and sum(c["phase"] == "warmup_excluded" for c in calls) == 1, "Grader warmup coverage")
        scored = [c for c in calls if c["phase"] == "measured_warm"]
        measured[backend] = {c["id"]: c for c in scored}
        check(len(scored) == 30 and set(measured[backend]) == set(pairs), "Grader measured coverage")
        tn = fp = fn = tp = 0
        for c in scored:
            check(c["status"] == "PASS" and c["expected"] == pairs[c["id"]]["expected"], "Grader status/label")
            actual_yes = pairs[c["id"]]["expected"] == "yes"
            check(c["decision"] in ("yes", "no"), "Grader contract")
            predicted_yes = c["decision"] == "yes"
            if backend == "laya":
                check(0 <= c["p_relevant"] <= 1, "Noul probability")
                check(predicted_yes == (c["p_relevant"] >= metrics["threshold"]), "Noul threshold")
            tn += not actual_yes and not predicted_yes
            fp += not actual_yes and predicted_yes
            fn += actual_yes and not predicted_yes
            tp += actual_yes and predicted_yes
        m = metrics[f"{backend}_metrics"]
        check([[tn, fp], [fn, tp]] == m["confusion_matrix"], "Confusion matrix")
        for key, value in {"precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn), "accuracy": (tp + tn) / 30}.items():
            check(close(value, m[key]), f"Metric {key}")
        times = sorted(c["elapsed_ms"] for c in scored)
        observed = m["latency_ms"]
        check(close(statistics.median(times), observed["median"]), "Grader median")
        check(close(statistics.mean(times), observed["mean"]), "Grader mean")
        check(close(times[math.ceil(len(times) * .95) - 1], observed["p95_nearest_rank"]), "Grader p95")
    with (HERE / "c1/results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == 30 and {r["id"] for r in rows} == set(pairs), "C1 CSV coverage")
    for r in rows:
        check(r["expected"] == pairs[r["id"]]["expected"], "CSV label")
        for backend in ("llm", "laya"):
            c = measured[backend][r["id"]]
            check(r[f"{backend}_decision"] == c["decision"], "CSV decision")
            check(float(r[f"{backend}_elapsed_ms"]) == c["elapsed_ms"], "CSV latency")

    c2 = load("c2/runs.json")
    check(c2["functional_runs_passed"] == 6 and not c2["automatic_original_harness_all_green"], "C2 status distinction")
    runs = {r["run"]: r for r in c2["runs"]}
    check(set(runs) == {f"S{i}_{b}" for i in (1, 2, 3) for b in ("llm", "laya")}, "C2 run coverage")
    for name, r in runs.items():
        check(r["raw_harness_status"] == "FAIL" and r["functional_status"] == "PASS", "Raw failures must be preserved")
        check(r["native_graph_completed"] and not r["native_runtime_errors"], "Native completion")
        check(r["raw_failure"]["type"] == "AssertionError" and r["embedding_actual_forward_count"] is None, "Observer limitation")
        check(sum(r["llm_by_role"].values()) == r["llm_calls"], "LLM role count")
        check(r["laya_predictions"] == r["laya_actual_forwards"], "Noul counts")
        check(r["initial_route"] == "vectorstore", "Initial route")
        for f in r["filtering"]:
            check(len(f["decisions"]) == f["input_count"] and sum(x == "yes" for x in f["decisions"]) == f["kept_count"], "Filtering decisions")
            check(f["expected_kept_sha256"] == f["actual_kept_sha256"], "Kept document hashes")
        if name.startswith("S3"):
            check(r["final_retries"] == 2 and r["final_datasource"] == "web_search", "Recovery budget/source")
            check(r["transitions"].count("rewrite_query") == 2 and all(f["kept_count"] == 0 for f in r["filtering"]), "Recovery filtering/rewrite")
        else:
            check(r["final_retries"] == 0 and r["final_datasource"] == "vectorstore", "Normal source/retries")
    check(sum(r["llm_calls"] for n, r in runs.items() if n.endswith("_llm")) == 28, "LLM total")
    check(sum(r["llm_calls"] for n, r in runs.items() if n.endswith("_laya")) == 11, "Hybrid LLM total")
    check(sum(r["laya_actual_forwards"] for r in runs.values()) == 17, "Noul total")
    for case in ("S1", "S2", "S3"):
        check(runs[f"{case}_llm"]["retrievals"] == runs[f"{case}_laya"]["retrievals"], "Paired retrievals")
    with (HERE / "c2/summary.csv").open(newline="") as f:
        csv_runs = list(csv.DictReader(f))
    check(len(csv_runs) == 6 and {r["run"] for r in csv_runs} == set(runs), "C2 CSV coverage")
    for r in csv_runs:
        check(r["raw_harness_status"] == "FAIL" and float(r["end_to_end_ms"]) == runs[r["run"]]["end_to_end_ms"], "C2 CSV status/timing")
    print(f"PASS: {len(listed)} public-file hashes; frozen/source hashes; C0/C1 metric recomputation; C2 record consistency.")
    print("Offline evidence verification only. Original C2 harness remains FAIL; embedding forward count UNVERIFIED.")


if __name__ == "__main__":
    verify()
