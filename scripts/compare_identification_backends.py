#!/usr/bin/env python3
"""Compare MuJoCo and Newton bags of an existing identification dataset.

``generate_identification_dataset.py`` already does this at the end of a run;
this script re-checks a written dataset without simulating anything, e.g. to
try other limits.  Limits come from the YAML config's ``comparison`` block
unless overridden.

Examples
--------
    python scripts/compare_identification_backends.py data/identification/kuka_lbr_iiwa_14_r820_table.csv
    python scripts/compare_identification_backends.py dataset.csv --q-link-rms 5e-4 --no-save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.backend_comparison import compare_backends, format_report
from elastic_sim.dataset import DEFAULT_CONFIG, comparison_report_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="Dataset CSV written by generate_identification_dataset.py")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML providing the comparison limits")
    parser.add_argument("--q-link-rms", type=float, default=None, help="Limit on link position RMS [rad]")
    parser.add_argument("--ft-relative-rms", type=float, default=None, help="Limit on relative link torque RMS")
    parser.add_argument("--deflection-relative-rms", type=float, default=None,
                        help="Limit on relative transmission deflection RMS")
    parser.add_argument("--output", default=None, help="Report CSV (default: next to the dataset)")
    parser.add_argument("--no-save", action="store_true", help="Print the report only")
    args = parser.parse_args()

    config_path = Path(args.config)
    thresholds = load_config(config_path if config_path.is_absolute() else _REPO / config_path).comparison
    overrides = {name: getattr(args, name) for name in ("q_link_rms", "ft_relative_rms", "deflection_relative_rms")}
    thresholds = replace(thresholds, **{name: value for name, value in overrides.items() if value is not None})

    csv_path = Path(args.dataset).expanduser().resolve()
    frame = pd.read_csv(csv_path)
    manifest_path = csv_path.with_suffix(".manifest.json")
    records = []
    if manifest_path.is_file():
        records = json.loads(manifest_path.read_text(encoding="utf-8")).get("records", [])
    else:
        print(f"no manifest at {manifest_path}; solver names will be unknown")

    report = compare_backends(frame, records, thresholds)
    print(format_report(report, thresholds))
    if args.no_save or report.empty:
        return
    output = Path(args.output).expanduser().resolve() if args.output else comparison_report_path(csv_path)
    report.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
