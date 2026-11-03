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
"""

import sys

from src.audit import cpcb_audit, cpcb_timestamp_check, station_audit
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
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        print(__doc__)
        sys.exit(1)
    STEPS[sys.argv[1]]()
