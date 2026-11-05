"""Small validation-year tuning of the LSTM before the main runs (Stage 6, D048).

Each candidate in ``nets.tuning`` (forecast.yaml) overrides the base settings; the LSTM is
trained on the training years (setting a, Tier B, seed 42) with early stopping on the
validation year, and scored on the whole validation year (2024) at the reported horizons,
beside persistence on the same rows. The test year is not used. The chosen candidate is
written into ``nets`` by hand and applies to both the LSTM and the graph network.

Output:
    reports/tables/forecast_nets_tuning.csv

Usage:
    python -m src.forecast.tune_nets
"""

import time

import numpy as np
import pandas as pd
import torch

from src.forecast.features import HourlyArrays, last_valid
from src.forecast.hourly_data import paths
from src.forecast.nets import Forecaster, net_arrays
from src.forecast.run_nets import DEV, Batches, predict, train_one
from src.forecast.run_tabular import issue_masks
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_TABLES

log = get_logger("forecast_tune")


def r2(y, p):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main(version: int = 1) -> None:
    cfg = load_config("forecast")
    hp, sp = paths(version)
    A = HourlyArrays.load(pd.read_parquet(hp), pd.read_parquet(sp), cfg)
    masks = issue_masks(A.times, cfg["splits"])
    H = cfg["max_horizon"]
    last24, _ = last_valid(A.var["pm25"], 24)
    rows = []
    for name, over in cfg["nets"]["tuning"].items():
        nc = {**cfg["nets"], **over}
        L = nc["input_hours"]
        arr = net_arrays(A, cfg, masks["train"], "a", "B")
        data = Batches(arr, L, H)
        tr = np.nonzero(masks["train"] & (np.arange(len(A.times)) >= L - 1))[0]
        va_all = np.nonzero(masks["val"])[0]
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        model = Forecaster(arr.past.shape[-1], arr.future.shape[-1], arr.static.shape[-1], H, nc["hidden"],
                           nc["layers"], nc["head_hidden"], nc["dropout"]).to(DEV)
        t0 = time.time()
        hist = train_one(model, data, tr, va_all[::6], nc, 42)
        p, _ = predict(model, data, va_all, arr, cfg["horizons"])
        p["persistence"] = last24[p["station_idx"], p["issue_idx"]]
        for h, g in p.groupby("h"):
            g = g[np.isfinite(g["persistence"])]
            rows.append({"candidate": name, **over, "h": h, "n": len(g), "r2_val": r2(g["y"], g["pred"]),
                         "r2_val_persistence": r2(g["y"], g["persistence"]), "epochs": len(hist),
                         "seconds": round(time.time() - t0)})
        log.info("%s: %s", name, {r["h"]: round(r["r2_val"], 3) for r in rows if r["candidate"] == name})
        del data
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_TABLES / "forecast_nets_tuning.csv", index=False)
    log.info("\n%s", out.pivot(index="candidate", columns="h", values="r2_val").round(3).to_string())


if __name__ == "__main__":
    main()
