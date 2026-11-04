"""Single entry point for pipeline steps.

Usage:
    python -m src.run <step>

Steps:
    boundaries   download GADM state boundaries
    openaq       OpenAQ locations -> sensor table -> daily aggregates
    cpcb         CPCB station list -> coordinates; collect hourly files from Downloads -> daily table
    audit        OpenAQ and CPCB audit tables, maps and summaries (needs the steps above)
    stage1       all of the above, in order
    tscheck      CPCB timestamp convention check against OpenAQ hourly (decision D024)
    ground       standardise CPCB + OpenAQ-only PM2.5 into one hourly table
    gee          Earth Engine extraction at stations (resumable; rerun to retry failed chunks)
    stage2       tscheck, ground, gee
    stage3       exploratory table, missingness, PM2.5 patterns, data-quality candidates
    qcoptions    retention of candidate QC rules and satellite thresholds (Stage 4 proposal)
    stage4       PM2.5 QC, targets, reanalysis daily, dataset v1, target/buffer comparison
    stage5       baselines under all splits + ablation, then results tables and figures
"""

import sys

from src.audit import cpcb_audit, cpcb_timestamp_check, station_audit
from src.align import reanalysis, targets
from src.clean import pm25_qc, qc_options
from src.eda import build_daily, data_quality, missingness, pm25_patterns
from src.eval import report, run_experiments
from src.features import build_dataset, compare
from src.ingest import (boundaries, cpcb_hourly, cpcb_stations, gee_extract, ground_pm25, openaq_daily,
                        openaq_inventory, openaq_locations)


def run_openaq() -> None:
    openaq_locations.main()
    openaq_inventory.main()
    openaq_daily.main()


def run_cpcb() -> None:
    cpcb_stations.main()
    cpcb_hourly.main()


def run_audit() -> None:
    station_audit.main()
    cpcb_audit.main()


STEPS = {
    "boundaries": boundaries.main,
    "openaq": run_openaq,
    "cpcb": run_cpcb,
    "audit": run_audit,
    "stage1": lambda: (boundaries.main(), run_openaq(), run_cpcb(), run_audit()),
    "tscheck": cpcb_timestamp_check.main,
    "ground": ground_pm25.main,
    "gee": gee_extract.main,
    "stage2": lambda: (cpcb_timestamp_check.main(), ground_pm25.main(), gee_extract.main()),
    "stage3": lambda: (build_daily.main(), missingness.main(), pm25_patterns.main(), data_quality.main()),
    "qcoptions": qc_options.main,
    "stage4": lambda: (pm25_qc.main(), targets.main(), reanalysis.main(), build_dataset.main(), compare.main()),
    "stage5": lambda: (run_experiments.main(), report.main()),
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        print(__doc__)
        sys.exit(1)
    STEPS[sys.argv[1]]()
