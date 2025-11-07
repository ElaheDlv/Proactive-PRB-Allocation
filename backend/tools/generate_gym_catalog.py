#!/usr/bin/env python3
"""
Generate episodic configs for the Gym PRB allocator.

Usage example:
python backend/tools/generate_gym_catalog.py \
  --output backend/assets/episodes/gym_full_catalog.json
"""

import argparse
import csv
import json
import math
import os
from itertools import product
from pathlib import Path
from typing import List

import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PRB Gym episode catalog.")
    parser.add_argument("--output", required=True, help="Destination JSON file.")
    parser.add_argument(
        "--trace-root",
        default="backend/assets/traces",
        help="Base directory where trace files live.",
    )
    parser.add_argument(
        "--duration-fallback",
        type=int,
        default=600,
        help="Fallback duration (decision steps) when trace metadata is missing.",
    )
    return parser.parse_args()


def trace_duration_seconds(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        last_time = 0.0
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            for row in reader:
                if not row:
                    continue
                try:
                    last_time = float(row[0])
                except ValueError:
                    continue
        if last_time <= 0:
            return 0.0
        if last_time >= 1e4:
            return last_time / 1000.0  # assume ms
        return last_time
    except Exception:
        return 0.0


def duration_to_steps(duration_s: float) -> int:
    sim_step = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0))
    decision_period = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))
    seconds_per_decision = sim_step * decision_period
    if seconds_per_decision <= 0:
        seconds_per_decision = 1.0
    return max(1, math.ceil(duration_s / seconds_per_decision))


def main():
    args = parse_args()

    ue_counts = [1, 3, 5, 7, 9, 11, 13]
    prb_choices = [1, 3, 9, 27, 81]
    std1_vals = [1, 3, 5, 7, 9, 11, 13]       # eMBB variability
    std2_vals = [0.1, 0.3, 0.5, 0.9]          # URLLC variability
    alpha_vals = [0.25, 0.5, 1, 1.5, 2]       # optional trace label
    beta_vals = [0.25, 0.5, 1, 1.5, 2]        # optional trace label

    trace_root = Path(args.trace_root)
    episodes = []
    counter = 0

    combos = product(
        ue_counts,
        ue_counts,
        prb_choices,
        prb_choices,
        std1_vals,
        std2_vals,
        alpha_vals,
        beta_vals,
    )

    for (
        ue_e,
        ue_u,
        prb_e,
        prb_u,
        std1,
        std2,
        alpha,
        beta,
    ) in combos:
        embb_trace = trace_root / f"embb_std1_{std1}_alpha{alpha}.csv"
        urllc_trace = trace_root / f"urllc_std2_{std2}_beta{beta}.csv"
        duration_e = trace_duration_seconds(embb_trace)
        duration_u = trace_duration_seconds(urllc_trace)
        duration_s = max(duration_e, duration_u)
        if duration_s <= 0:
            duration_steps = args.duration_fallback
        else:
            duration_steps = duration_to_steps(duration_s)

        episodes.append(
            {
                "id": f"ep{counter:05d}",
                "duration_steps": duration_steps,
                "initial_prb": {"eMBB": prb_e, "URLLC": prb_u},
                "slices": {
                    "eMBB": {
                        "ue_count": ue_e,
                        "trace": str(embb_trace),
                        "trace_speedup": 1.0,
                    },
                    "URLLC": {
                        "ue_count": ue_u,
                        "trace": str(urllc_trace),
                        "trace_speedup": 1.0,
                    },
                },
            }
        )
        counter += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"episodes": episodes}, fp, indent=2)
    print(f"Wrote {len(episodes)} episodes to {output_path}")


if __name__ == "__main__":
    main()
