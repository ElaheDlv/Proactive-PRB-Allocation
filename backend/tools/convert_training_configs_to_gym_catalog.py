#!/usr/bin/env python3
"""
Convert legacy xApp DQN training configs into the episodic format required by
the Gym-style PRB allocator.

Example:
python backend/tools/convert_training_configs_to_gym_catalog.py \
  --input backend/notebooks/xapp_dqn_training_configs.json \
  --trace-root backend/notebooks/Unified_CMTC/traces/aligned \
  --output backend/assets/episodes/gym_from_training_config.json
"""

import argparse
import csv
import json
import math
from pathlib import Path

import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Convert training configs into Gym episode catalog.")
    parser.add_argument("--input", required=True, help="Path to xapp_dqn_training_configs.json")
    parser.add_argument("--output", required=True, help="Destination JSON for the Gym catalog")
    parser.add_argument(
        "--trace-root",
        required=True,
        help="Directory where trace CSVs live (filenames from the config are resolved relative to this path)",
    )
    parser.add_argument(
        "--duration-fallback",
        type=int,
        default=600,
        help="Fallback decision-step length if trace duration cannot be determined",
    )
    parser.add_argument(
        "--trace-speedup",
        type=float,
        default=1.0,
        help="Optional speedup factor to apply to all traces (mirrors TRACE_SPEEDUP)",
    )
    return parser.parse_args()


def trace_duration_seconds(path: Path) -> float:
    """Return the final timestamp in seconds (auto-detect ms scale)."""
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
        if last_time <= 0.0:
            return 0.0
        if last_time >= 1e4:  # assume milliseconds
            return last_time / 1000.0
        return last_time
    except Exception:
        return 0.0


def duration_steps(duration_s: float) -> int:
    sim_step = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0))
    decision_period = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))
    seconds_per_decision = max(1e-9, sim_step * decision_period)
    return max(1, math.ceil(duration_s / seconds_per_decision))


def main():
    args = parse_args()
    input_path = Path(args.input)
    trace_root = Path(args.trace_root)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as fp:
        configs = json.load(fp)

    episodes = []
    for idx, cfg in enumerate(configs):
        ue_count = int(cfg.get("num_ues_per_slice", 1))
        prb_embb = int(cfg.get("embb_default_prb", 0))
        prb_urllc = int(cfg.get("urllc_default_prb", 0))
        trace_embb = trace_root / cfg.get("trace_files", {}).get("embb", "")
        trace_urllc = trace_root / cfg.get("trace_files", {}).get("urllc", "")

        duration_e = trace_duration_seconds(trace_embb)
        duration_u = trace_duration_seconds(trace_urllc)
        duration_s = max(duration_e, duration_u)
        if duration_s <= 0.0:
            steps = args.duration_fallback
        else:
            steps = duration_steps(duration_s / max(1e-9, args.trace_speedup))

        episodes.append(
            {
                "id": f"cfg_{idx:05d}",
                "duration_steps": int(steps),
                "freeze_mobility": True,
                "initial_prb": {"eMBB": prb_embb, "URLLC": prb_urllc},
                "slices": {
                    "eMBB": {
                        "ue_count": ue_count,
                        "trace": str(trace_embb),
                        "trace_speedup": args.trace_speedup,
                    },
                    "URLLC": {
                        "ue_count": ue_count,
                        "trace": str(trace_urllc),
                        "trace_speedup": args.trace_speedup,
                    },
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"episodes": episodes}, fp, indent=2)
    print(f"Wrote {len(episodes)} episodes to {output_path}")


if __name__ == "__main__":
    main()
