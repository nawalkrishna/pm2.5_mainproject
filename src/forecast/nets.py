"""LSTM and graph-network forecast models and their input arrays (Stage 6, D048, D051).

Both models read the past ``input_hours`` of each station's hourly inputs and predict PM2.5 at
1..max_horizon hours ahead at once, with a head that sees the encoder state, the station's
static features, the target hour's calendar, a horizon embedding and, in setting b, the
reanalysis weather at the target hour. The head predicts the change from the station's latest
(filled) PM2.5 value, i.e. a residual on persistence: without it the first runs fell well
below persistence at short horizons (1 h R² 0.63 against 0.78).

    LSTM      one shared encoder, applied to each station's own series
    GC-LSTM   at every time step a graph convolution mixes each station's inputs with its
              neighbours' (stations within graph_km, Gaussian distance weights, symmetric
              normalisation, Qi et al. 2019), then the shared LSTM encoder

Inputs follow the issue-time rules of D046: instantaneous fields at hour t, averaged ones
(precipitation, MERRA-2) at t - 1, satellite values from the latest satellite day. Missing
inputs (D051): PM2.5 carried forward up to pm_ffill_hours, else the station's training
median, with an observed mask; all inputs standardised with training statistics; NaN -> 0.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.align.spatial import haversine_m
from src.forecast.features import HourlyArrays, lag, last_valid, satellite_features


def _std(a: np.ndarray, train_cols: np.ndarray) -> np.ndarray:
    ref = a[..., train_cols] if a.ndim == 2 else a
    mu, sd = np.nanmean(ref), np.nanstd(ref)
    return np.nan_to_num((a - mu) / (sd if sd > 0 else 1.0)).astype(np.float32)


def _cal(times: pd.DatetimeIndex) -> np.ndarray:
    h, d, w = times.hour.to_numpy(), times.day_of_year.to_numpy(), times.day_of_week.to_numpy()
    yl = np.where(times.is_leap_year, 366, 365)
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * d / yl),
                     np.cos(2 * np.pi * d / yl), np.sin(2 * np.pi * w / 7), np.cos(2 * np.pi * w / 7)], 1).astype(np.float32)


@dataclass
class NetArrays:
    past: np.ndarray        # (N, T, F_past)
    future: np.ndarray      # (N, T, F_future) known at the target hour (calendar; + weather in setting b)
    static: np.ndarray      # (N, F_static)
    y: np.ndarray           # (N, T) standardised PM2.5, NaN where missing
    tier_ok: np.ndarray     # (N, T) issue hours where this tier applies (D049)
    y_mean: float
    y_sd: float


def net_arrays(A: HourlyArrays, cfg: dict, train: np.ndarray, setting: str, tier: str) -> NetArrays:
    pm = A.var["pm25"]
    obs = np.isfinite(pm)
    y_mean, y_sd = float(np.nanmean(pm[:, train])), float(np.nanstd(pm[:, train]))
    med = np.nanmedian(np.where(train[None, :], pm, np.nan), axis=1)
    med = np.where(np.isfinite(med), med, np.nanmedian(pm[:, train]))
    filled, _ = last_valid(pm, cfg["nets"]["pm_ffill_hours"])
    filled = np.where(np.isfinite(filled), filled, med[:, None])
    chans = [((filled - y_mean) / y_sd).astype(np.float32), obs.astype(np.float32)]
    lag_avg = cfg["timing"]["average_lag"]
    chans += [_std(A.var[v], train) for v in cfg["instant_met"]]
    chans += [_std(lag(A.var[v], lag_avg), train) for v in cfg["average_met"] + cfg["aerosol"]]
    cal = _cal(A.times)
    chans += [np.broadcast_to(cal[None, :, i], pm.shape) for i in range(cal.shape[1])]
    sat = satellite_features(A, cfg)
    if tier == "A":
        for k, v in sat.items():
            chans.append(_std(v, train))
        chans += [np.isfinite(sat["sat_aod_clear"]).astype(np.float32), np.isfinite(sat["sat_no2_trop"]).astype(np.float32)]
    past = np.stack(chans, -1)
    fut = [np.broadcast_to(cal[None, :, i], pm.shape) for i in range(cal.shape[1])]
    if setting == "b":
        fut += [_std(A.var[v], train) for v in cfg["instant_met"] + cfg["average_met"]]
    future = np.stack(fut, -1).astype(np.float32)
    st = A.static.set_index("station_id").loc[A.stations["station_id"], [c for c in cfg["static"] if c != "population_b5"]]
    st = st.assign(population_b5=np.nanmean(A.var["population_b5"], 1))
    static = np.nan_to_num(((st - st.mean()) / st.std().replace(0, 1)).to_numpy(np.float32))
    tier_ok = np.ones(pm.shape, bool) if tier == "B" else np.isfinite(sat["sat_aod_clear"])
    return NetArrays(past, future, static, ((pm - y_mean) / y_sd).astype(np.float32), tier_ok, y_mean, y_sd)


def graph_matrix(stations: pd.DataFrame, km: float) -> np.ndarray:
    """Symmetric-normalised adjacency with self-loops: w = exp(-(d / (km / 2))^2) for d <= km."""
    lat, lon = stations["latitude"].to_numpy(), stations["longitude"].to_numpy()
    d = np.stack([haversine_m(lat[i], lon[i], lat, lon) / 1000 for i in range(len(lat))])
    w = np.where(d <= km, np.exp(-(d / (km / 2)) ** 2), 0.0)
    np.fill_diagonal(w, 1.0)
    dinv = 1 / np.sqrt(w.sum(1))
    return (w * dinv[:, None] * dinv[None, :]).astype(np.float32)


class Forecaster(nn.Module):
    def __init__(self, f_past: int, f_future: int, f_static: int, n_h: int, hidden: int, layers: int,
                 head_hidden: int, dropout: float, adj: np.ndarray | None = None):
        super().__init__()
        self.adj = None if adj is None else nn.Parameter(torch.from_numpy(adj), requires_grad=False)
        enc_in = f_past
        if self.adj is not None:
            self.gc = nn.Linear(f_past, hidden)
            enc_in = f_past + hidden
        self.lstm = nn.LSTM(enc_in, hidden, num_layers=layers, batch_first=True)
        self.h_emb = nn.Embedding(n_h, 8)
        self.head = nn.Sequential(nn.Linear(hidden + f_static + f_future + 8, head_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head_hidden, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, 1))

    def forward(self, past, future, static):
        """past (B, N, L, F), future (B, N, H, Ff), static (N, Fs) -> (B, N, H)."""
        b, n, L, f = past.shape
        if self.adj is not None:
            mixed = torch.einsum("ij,bjlf->bilf", self.adj, past)
            past = torch.cat([past, torch.relu(self.gc(mixed))], -1)
        _, (hT, _) = self.lstm(past.reshape(b * n, L, -1))
        z = hT[-1].reshape(b, n, 1, -1)
        H = future.shape[2]
        e = self.h_emb.weight[None, None, :H].expand(b, n, H, -1)
        s = static[None, :, None, :].expand(b, n, H, -1)
        x = torch.cat([z.expand(b, n, H, -1), s, future, e], -1)
        last = past[:, :, -1, 0:1]                       # standardised PM2.5 at the issue hour (filled)
        return last + self.head(x).squeeze(-1)           # residual on persistence
