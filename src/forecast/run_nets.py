"""Train and evaluate the LSTM and graph-network forecasters (Stage 6, D047-D051).

For each seed x setting (a, b) x tier (B, A) x model (lstm, gcn): train on issue hours in the
training years (each epoch a random epoch_fraction of them, for speed) with early stopping on
the validation year (every 6th issue hour), then predict 1..max_horizon hours ahead from every
issue hour of the test year. The loss and the evaluation use only observed targets at issue
hours where the tier applies (D049). Seeds run outermost, so every configuration has seed 42
first. Each run is saved to data/processed/forecast_nets_parts/ and skipped if already there
(resumable); the combined files are written at the end.

Outputs:
    data/processed/forecast_predictions_nets_v<version>.parquet   reported horizons, one column
                                                                  per <model>_<setting>_<tier>_s<seed>
    reports/tables/forecast_nets_by_horizon.csv                   RMSE and R² at every horizon
    reports/tables/forecast_nets_training.csv                     epochs, losses, time

Usage:
    python -m src.forecast.run_nets [--models lstm gcn] [--settings a b] [--tiers B A] [--seeds 42]
    python -m src.forecast.run_nets --combine        # only rebuild the combined files from the parts
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.forecast.features import HourlyArrays
from src.forecast.hourly_data import paths
from src.forecast.nets import Forecaster, graph_matrix, net_arrays
from src.forecast.run_tabular import issue_masks
from src.ingest.ground_pm25 import STATIONS_PATH
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PROCESSED, REPORTS_TABLES
from src.utils.run_log import log_run

log = get_logger("forecast_nets")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH = 16
PARTS = DATA_PROCESSED / "forecast_nets_parts"


class Batches:
    """Gathers (past, future, target, mask) for a batch of issue hours, on the device."""

    def __init__(self, arr, L: int, H: int):
        self.past = torch.from_numpy(np.ascontiguousarray(arr.past)).to(DEV)
        self.future = torch.from_numpy(np.ascontiguousarray(arr.future)).to(DEV)
        self.static = torch.from_numpy(arr.static).to(DEV)
        self.y = torch.from_numpy(arr.y).to(DEV)
        self.ok = torch.from_numpy(arr.tier_ok).to(DEV)
        self.L, self.H, self.T = L, H, arr.y.shape[1]

    def get(self, idx: np.ndarray):
        i = torch.from_numpy(idx).to(DEV)
        win = i[:, None] + torch.arange(-self.L + 1, 1, device=DEV)
        tgt = i[:, None] + torch.arange(1, self.H + 1, device=DEV)
        inside = tgt < self.T
        tgt = tgt.clamp(max=self.T - 1)
        past = self.past[:, win].permute(1, 0, 2, 3)
        fut = self.future[:, tgt].permute(1, 0, 2, 3)
        y = self.y[:, tgt].permute(1, 0, 2)
        m = torch.isfinite(y) & inside[:, None, :] & self.ok[:, i].T[:, :, None]
        return past, fut, torch.nan_to_num(y), m


def masked_mse(p, y, m):
    return ((p - y) ** 2 * m).sum() / m.sum().clamp(min=1)


def train_one(model, data: Batches, tr_idx, va_idx, nc: dict, seed: int):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=nc["learning_rate"], weight_decay=nc.get("weight_decay", 0.0))
    best, best_state, bad, hist = np.inf, None, 0, []
    per_epoch = int(len(tr_idx) * nc["epoch_fraction"])
    for epoch in range(nc["max_epochs"]):
        t0 = time.time()
        model.train()
        order = rng.permutation(tr_idx)[:per_epoch]
        tot = 0.0
        for k in range(0, len(order), nc["batch_size"]):
            past, fut, y, m = data.get(order[k:k + nc["batch_size"]])
            loss = masked_mse(model(past, fut, data.static), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            se = n = 0.0
            for k in range(0, len(va_idx), EVAL_BATCH):
                past, fut, y, m = data.get(va_idx[k:k + EVAL_BATCH])
                se += float((((model(past, fut, data.static) - y) ** 2) * m).sum())
                n += float(m.sum())
        val = se / max(n, 1)
        hist.append({"epoch": epoch, "train_loss": tot / max(1, len(order) // nc["batch_size"]), "val_loss": val,
                     "seconds": time.time() - t0})
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= nc["patience"]:
                break
    model.load_state_dict(best_state)
    return hist


def predict(model, data: Batches, te_idx, arr, horizons):
    """Reported-horizon predictions (µg/m³) and per-horizon sums for RMSE / R² at every horizon."""
    rows, H = [], data.H
    sums = np.zeros((H, 4))                                  # n, sum y, sum y², sse
    model.eval()
    with torch.no_grad():
        for k in range(0, len(te_idx), EVAL_BATCH):
            idx = te_idx[k:k + EVAL_BATCH]
            past, fut, y, m = data.get(idx)
            p = model(past, fut, data.static) * arr.y_sd + arr.y_mean
            yy = y * arr.y_sd + arr.y_mean
            mf = m.float()
            sums[:, 0] += mf.sum((0, 1)).cpu().numpy()
            sums[:, 1] += (yy * mf).sum((0, 1)).cpu().numpy()
            sums[:, 2] += (yy ** 2 * mf).sum((0, 1)).cpu().numpy()
            sums[:, 3] += (((p - yy) ** 2) * mf).sum((0, 1)).cpu().numpy()
            for h in horizons:
                b, s = torch.nonzero(m[:, :, h - 1], as_tuple=True)
                rows.append(pd.DataFrame({"station_idx": s.cpu().numpy(), "issue_idx": idx[b.cpu().numpy()],
                                          "h": h, "y": yy[b, s, h - 1].cpu().numpy(),
                                          "pred": p[b, s, h - 1].cpu().numpy()}))
    return pd.concat(rows, ignore_index=True), sums


def combine(A: HourlyArrays, version: int = 1) -> None:
    """Join the saved runs into the combined prediction, horizon and training files."""
    names = sorted(f.stem[len("pred_"):] for f in PARTS.glob("pred_*.parquet"))
    preds = [pd.read_parquet(PARTS / f"pred_{n}.parquet").set_index(["station_idx", "issue_idx", "h"]) for n in names]
    out = pd.concat([d.drop(columns="y") for d in preds], axis=1)
    ys = pd.concat([d["y"] for d in preds])
    out["y"] = ys[~ys.index.duplicated()].reindex(out.index)
    out = out.reset_index()
    out["station_id"] = A.stations["station_id"].to_numpy()[out["station_idx"]]
    out["time_issue"] = A.times[out["issue_idx"].to_numpy()]
    path = DATA_PROCESSED / f"forecast_predictions_nets_v{version}.parquet"
    out.drop(columns="station_idx").to_parquet(path, index=False)
    pd.concat([pd.read_csv(PARTS / f"curve_{n}.csv") for n in names]).to_csv(
        REPORTS_TABLES / "forecast_nets_by_horizon.csv", index=False)
    pd.concat([pd.read_csv(PARTS / f"hist_{n}.csv") for n in names]).to_csv(
        REPORTS_TABLES / "forecast_nets_training.csv", index=False)
    log.info("Combined %d runs, %d rows -> %s", len(names), len(out), path)


def main(models=None, settings=None, tiers=None, seeds=None, version: int = 1, only_combine: bool = False) -> None:
    cfg = load_config("forecast")
    nc = cfg["nets"]
    models, settings, tiers = models or ["lstm", "gcn"], settings or ["a", "b"], tiers or ["B", "A"]
    seeds = seeds or nc["seeds"]
    hourly_path, sat_path = paths(version)
    A = HourlyArrays.load(pd.read_parquet(hourly_path), pd.read_parquet(sat_path), cfg)
    if only_combine:
        return combine(A, version)
    masks = issue_masks(A.times, cfg["splits"])
    L, H = nc["input_hours"], cfg["max_horizon"]
    ok_window = np.arange(len(A.times)) >= L - 1
    tr_idx = np.nonzero(masks["train"] & ok_window)[0][:: nc["train_issue_stride"]]
    va_idx = np.nonzero(masks["val"])[0][::6]
    te_idx = np.nonzero(masks["test"])[0]
    coords = pd.read_csv(STATIONS_PATH).set_index("station_id").loc[A.stations["station_id"], ["latitude", "longitude"]]
    adj = graph_matrix(coords.reset_index(), nc["graph_km"])
    log.info("Device %s; train %d, val %d, test %d issue hours; graph edges per station (median) %d", DEV,
             len(tr_idx), len(va_idx), len(te_idx), int(np.median((adj > 0).sum(1)) - 1))
    PARTS.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for setting in settings:
            for tier in tiers:
                todo = [m for m in models if not (PARTS / f"pred_{m}_{setting}_{tier}_s{seed}.parquet").exists()]
                if not todo:
                    continue
                arr = net_arrays(A, cfg, masks["train"], setting, tier)
                data = Batches(arr, L, H)
                for model_name in todo:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    torch.backends.cudnn.deterministic = True
                    model = Forecaster(arr.past.shape[-1], arr.future.shape[-1], arr.static.shape[-1], H, nc["hidden"],
                                       nc["layers"], nc["head_hidden"], nc["dropout"],
                                       adj if model_name == "gcn" else None).to(DEV)
                    name = f"{model_name}_{setting}_{tier}_s{seed}"
                    t0 = time.time()
                    hist = train_one(model, data, tr_idx, va_idx, nc, seed)
                    p, sums = predict(model, data, te_idx, arr, cfg["horizons"])
                    n, sy, syy, sse = sums.T
                    curve = pd.DataFrame({"config": name, "h": np.arange(1, H + 1), "n": n,
                                          "rmse": np.sqrt(sse / n), "r2": 1 - sse / (syy - sy ** 2 / n)})
                    curve.to_csv(PARTS / f"curve_{name}.csv", index=False)
                    pd.DataFrame(hist).assign(config=name).to_csv(PARTS / f"hist_{name}.csv", index=False)
                    p.rename(columns={"pred": name}).to_parquet(PARTS / f"pred_{name}.parquet", index=False)
                    log.info("%s: %d epochs, best val %.4f, %.0f s; R² at 1/24/72 h %.3f / %.3f / %.3f", name,
                             len(hist), min(x["val_loss"] for x in hist), time.time() - t0,
                             curve["r2"].iloc[0], curve["r2"].iloc[23], curve["r2"].iloc[H - 1])
                del data
                torch.cuda.empty_cache()
    combine(A, version)
    log_run("forecast_nets", config_hash(cfg), metrics={"seeds": seeds, "models": models})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*")
    p.add_argument("--settings", nargs="*")
    p.add_argument("--tiers", nargs="*")
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--combine", action="store_true")
    a = p.parse_args()
    main(a.models, a.settings, a.tiers, a.seeds, only_combine=a.combine)
