#!/usr/bin/env python3
"""Convert ADTC profiler submission.json telemetry into S_perf / S_eff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TPS_REFERENCE = 15.0
RAM_LIMIT_GB = 7.0
THERMAL_PENALTY = 10.0


def dig(obj: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    """Return the first present nested value among candidate paths."""
    for path in paths:
        cur: Any = obj
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            return cur
    return None


def as_gb(value: float | int | None, *, unit_hint: str | None = None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if unit_hint == "mb" or (unit_hint is None and v > 64):
        # Profiler commonly reports peak RSS in MB; values >> 64 are not GB for 8GB laptops.
        return v / 1024.0
    return v


def compute(report: dict[str, Any]) -> dict[str, Any]:
    tps = dig(
        report,
        ("throughput", "tokens_per_second_generation"),
        ("throughput", "tokens_per_second"),
        ("metrics", "throughput", "tokens_per_second_generation"),
        ("tps",),
    )
    peak_mb = dig(
        report,
        ("memory", "peak_rss_mb"),
        ("metrics", "memory", "peak_rss_mb"),
    )
    peak_gb_raw = dig(
        report,
        ("memory", "peak_rss_gb"),
        ("metrics", "memory", "peak_rss_gb"),
        ("peak_rss_gb",),
    )
    peak_gb = as_gb(peak_mb, unit_hint="mb") if peak_mb is not None else as_gb(peak_gb_raw)

    throttled = dig(
        report,
        ("thermal", "throttled"),
        ("thermals", "throttled"),
        ("metrics", "thermal", "throttled"),
    )
    peak_temp = dig(
        report,
        ("thermal", "peak_temp_c"),
        ("thermals", "peak_core_temp_c"),
        ("thermal", "max_temp_c"),
        ("metrics", "thermal", "peak_temp_c"),
    )

    if tps is None:
        raise SystemExit("Could not find throughput tokens/sec in report")
    if peak_gb is None:
        raise SystemExit("Could not find peak RSS in report")

    tps_f = float(tps)
    s_perf = min(tps_f / TPS_REFERENCE, 1.0) * 100.0
    s_eff = max(0.0, (RAM_LIMIT_GB - peak_gb) / RAM_LIMIT_GB) * 100.0

    thermal_hit = bool(throttled) or (
        peak_temp is not None and float(peak_temp) > 85.0
    )
    penalty = THERMAL_PENALTY if thermal_hit else 0.0
    # Accuracy is judged externally; mechanical subtotal excludes S_acc.
    mech = 0.30 * s_perf + 0.20 * s_eff - penalty

    return {
        "tps": tps_f,
        "peak_rss_gb": peak_gb,
        "s_perf": s_perf,
        "s_eff": s_eff,
        "thermal_penalty": penalty,
        "mechanical_subtotal_excl_sacc": mech,
        "measured_on": report.get("measured_on"),
        "team_id": report.get("team_id")
        or dig(report, ("metadata", "team_id")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=Path("submission.json"),
        help="Profiler JSON (default: submission.json)",
    )
    args = parser.parse_args()
    if not args.report.is_file():
        raise SystemExit(f"File not found: {args.report}")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    scores = compute(report)

    print("ADTC score breakdown (from profiler telemetry)")
    print(f"  measured_on : {scores['measured_on']}")
    print(f"  team_id     : {scores['team_id']}")
    print(f"  TPS         : {scores['tps']:.3f}  (ref {TPS_REFERENCE})")
    print(f"  peak RSS    : {scores['peak_rss_gb']:.3f} GB  (limit {RAM_LIMIT_GB})")
    print(f"  S_perf      : {scores['s_perf']:.2f}")
    print(f"  S_eff       : {scores['s_eff']:.2f}")
    print(f"  P_thermal   : {scores['thermal_penalty']:.0f}")
    print(
        f"  0.30*S_perf + 0.20*S_eff - P_thermal = "
        f"{scores['mechanical_subtotal_excl_sacc']:.2f}"
    )
    print("  (S_acc is 50% and is scored by judges — not in this file)")
    print()
    print("Devpost form (plain numbers):")
    print(f"  throughput score : {scores['s_perf']:.2f}")
    print(f"  efficiency score : {scores['s_eff']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
