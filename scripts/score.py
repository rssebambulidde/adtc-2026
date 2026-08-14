#!/usr/bin/env python3
"""Convert an adtc-profiler report into the normalized ADTC leaderboard scores.

Field names follow adtc-profiler.schema.json exactly. The profiler emits raw
telemetry; the Devpost form wants the normalized 0-100 numbers computed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TPS_REFERENCE = 15.0
RAM_LIMIT_GB = 7.0
THERMAL_PENALTY = 10.0
TEMP_LIMIT_C = 85.0


def require(obj: dict, key: str, ctx: str) -> object:
    if key not in obj:
        raise SystemExit(f"Report missing {ctx}.{key} — is this an adtc-profiler report?")
    return obj[key]


def compute(report: dict) -> dict:
    throughput = require(report, "throughput", "report")
    memory = require(report, "memory", "report")
    thermal = require(report, "cpu_thermal", "report")

    tps = float(require(throughput, "tokens_per_second_generation", "throughput"))
    peak_rss_gb = float(require(memory, "peak_rss_mb", "memory")) / 1024.0

    throttled = bool(thermal.get("throttled", False))
    peak_temp = thermal.get("core_temp_c_peak")
    over_temp = peak_temp is not None and float(peak_temp) > TEMP_LIMIT_C
    penalty = THERMAL_PENALTY if (throttled or over_temp) else 0.0

    s_perf = min(tps / TPS_REFERENCE, 1.0) * 100.0
    s_eff = max(0.0, (RAM_LIMIT_GB - peak_rss_gb) / RAM_LIMIT_GB) * 100.0

    accuracy = report.get("accuracy") or []
    environment = report.get("environment") or {}
    model_info = report.get("model_info") or {}

    return {
        "team_id": (report.get("submission") or {}).get("team_id"),
        "measured_on": environment.get("measured_on"),
        "tps": tps,
        "first_token_latency_ms": throughput.get("first_token_latency_ms"),
        "peak_rss_gb": peak_rss_gb,
        "steady_rss_gb": float(memory.get("steady_state_rss_mb", 0.0)) / 1024.0,
        "s_perf": s_perf,
        "s_eff": s_eff,
        "penalty": penalty,
        "throttled": throttled,
        "peak_temp_c": peak_temp,
        "weighted_known": 0.30 * s_perf + 0.20 * s_eff - penalty,
        "accuracy_entries": accuracy,
        "params_match": model_info.get("params_match"),
        "params_count": model_info.get("params_count"),
        "oom_risk": peak_rss_gb > RAM_LIMIT_GB,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=Path("submission.json"))
    args = parser.parse_args()
    if not args.report.is_file():
        raise SystemExit(f"File not found: {args.report}")

    s = compute(json.loads(args.report.read_text(encoding="utf-8")))

    print("ADTC score breakdown")
    print(f"  team_id      : {s['team_id']}")
    print(f"  measured_on  : {s['measured_on']}")
    print(f"  params_count : {s['params_count']}  (claim matches: {s['params_match']})")
    print()
    print(f"  TPS          : {s['tps']:.2f}   (reference {TPS_REFERENCE})")
    print(f"  TTFT         : {s['first_token_latency_ms']} ms")
    print(f"  peak RSS     : {s['peak_rss_gb']:.3f} GB  (budget {RAM_LIMIT_GB})")
    print(f"  steady RSS   : {s['steady_rss_gb']:.3f} GB")
    print(f"  throttled    : {s['throttled']}   peak temp: {s['peak_temp_c']}")
    print()
    print(f"  S_perf       : {s['s_perf']:.2f}   (weight 30%)")
    print(f"  S_eff        : {s['s_eff']:.2f}   (weight 20%)")
    print(f"  P_thermal    : {s['penalty']:.0f}")
    print(f"  known subtotal: {s['weighted_known']:.2f}  (excludes S_acc, the other 50%)")

    if s["accuracy_entries"]:
        print()
        print("  accuracy benchmarks run locally:")
        for a in s["accuracy_entries"]:
            print(
                f"    {a.get('benchmark')} ({a.get('language')}, n={a.get('samples')}): "
                f"{a.get('score')} {a.get('metric', '')}".rstrip()
            )
    else:
        print("\n  accuracy: none in report (judges score S_acc separately)")

    print()
    print("Devpost form — enter these plain numbers:")
    print(f"  S_perf: {s['s_perf']:.2f}")
    print(f"  S_eff : {s['s_eff']:.2f}")

    if s["oom_risk"]:
        print("\nWARNING: peak RSS exceeds the 7 GB budget — OOM means disqualification.")
    if s["params_match"] is False:
        print("\nWARNING: parameters_estimate in metadata.json disagrees with the GGUF header.")


if __name__ == "__main__":
    main()
