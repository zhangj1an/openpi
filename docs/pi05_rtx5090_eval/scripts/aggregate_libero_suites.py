"""Aggregate per-task LIBERO results.json files into one summary JSON + markdown tables."""

import json
import pathlib
import sys

import numpy as np

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
OPENPI_REPORTED = {"libero_spatial": 98.8, "libero_object": 98.2, "libero_goal": 98.0, "libero_10": 92.4}


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def main(root: str, out_json: str):
    root_p = pathlib.Path(root)
    summary = {"suites": {}}
    all_rt, all_srv, all_eps = [], [], []
    for suite in SUITES:
        eps, rt, srv, tasks = [], [], [], []
        for tid in range(10):
            r = json.loads((root_p / suite / f"task{tid}" / "results.json").read_text())
            assert len(r["episodes"]) == 50, (suite, tid, len(r["episodes"]))
            eps += r["episodes"]
            rt += r["raw"]["client_roundtrip"]
            srv += r["raw"]["server_policy_infer"]
            tasks.append({"task_id": tid, "task": r["episodes"][0]["task"],
                          "successes": int(sum(e["success"] for e in r["episodes"])), "episodes": 50})
        n_succ = int(sum(e["success"] for e in eps))
        summary["suites"][suite] = {
            "successes": n_succ,
            "episodes": len(eps),
            "success_rate_pct": 100.0 * n_succ / len(eps),
            "openpi_reported_pct": OPENPI_REPORTED[suite],
            "per_task": tasks,
            "mean_steps_success": float(np.mean([e["steps"] for e in eps if e["success"]])),
            "chunks": len(rt),
            "server_infer_ms": {"p50": pct(srv, 50), "p90": pct(srv, 90), "p99": pct(srv, 99)},
            "client_roundtrip_ms": {"p50": pct(rt, 50), "p90": pct(rt, 90), "p99": pct(rt, 99)},
            "failed_episodes": [(e["task_id"], e["episode"]) for e in eps if not e["success"]],
        }
        all_rt += rt
        all_srv += srv
        all_eps += eps
    n = int(sum(e["success"] for e in all_eps))
    summary["overall"] = {"successes": n, "episodes": len(all_eps), "success_rate_pct": 100.0 * n / len(all_eps),
                          "mean_of_suites_pct": float(np.mean([s["success_rate_pct"] for s in summary["suites"].values()])),
                          "openpi_reported_mean_pct": float(np.mean(list(OPENPI_REPORTED.values()))),
                          "chunks": len(all_rt),
                          "server_infer_ms": {"p50": pct(all_srv, 50), "p90": pct(all_srv, 90), "p99": pct(all_srv, 99)}}
    pathlib.Path(out_json).write_text(json.dumps(summary, indent=2))

    print("| Suite | Success | Rate | openpi reported | Per task (/50) |")
    print("| --- | ---: | ---: | ---: | --- |")
    for suite, s in summary["suites"].items():
        per = " ".join(str(t["successes"]) for t in s["per_task"])
        print(f"| {suite} | {s['successes']}/500 | {s['success_rate_pct']:.1f} % | {s['openpi_reported_pct']:.1f} % | {per} |")
    o = summary["overall"]
    print(f"| **All** | {o['successes']}/{o['episodes']} | {o['mean_of_suites_pct']:.2f} % | {o['openpi_reported_mean_pct']:.2f} % | |")
    for suite, s in summary["suites"].items():
        print(suite, "failed:", s["failed_episodes"], "server p50", round(s["server_infer_ms"]["p50"], 2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
