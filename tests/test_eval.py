import numpy as np
import pandas as pd

from src.eval import splits as S
from src.eval.metrics import metrics
from src.models.baselines import Climatology


def test_metrics_values():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
    assert m["bias"] == 1.0 and m["mae"] == 1.0 and m["rmse"] == 1.0 and np.isclose(m["r"], 1.0)
    assert np.isclose(m["r2"], 1 - 3 / 2)


def stations():
    # two stations 5 km apart (one block), one 300 km away
    return pd.DataFrame({"station_id": ["a", "b", "c"], "latitude": [12.97, 13.01, 15.6], "longitude": [77.59, 77.60, 77.6]})


def test_blocks_link_nearby_stations():
    b = S.spatial_blocks(stations(), 50)
    assert b["a"] == b["b"] != b["c"]


def test_spatial_folds_never_share_stations_or_blocks():
    df = pd.DataFrame({"station_id": list("aabbcc") * 2, "year": [2024] * 6 + [2025] * 6})
    blocks = S.spatial_blocks(stations(), 50)
    for f in S.blocked(df, blocks, 2) + S.loso(df) + S.combined(df, blocks, 2, [2024], [2025]):
        assert not set(df.loc[f.train, "station_id"]) & set(df.loc[f.test, "station_id"])
        assert not set(df.loc[f.train, "station_id"].map(blocks)) & set(df.loc[f.test, "station_id"].map(blocks)) \
            or f.split == "loso"
    comb = S.combined(df, blocks, 2, [2024], [2025])
    for f in comb:
        assert set(df.loc[f.train, "year"]) <= {2024} and set(df.loc[f.test, "year"]) <= {2025}
    t = S.temporal(df, [2024], [2025])[0]
    assert set(df.loc[t.train, "year"]) == {2024} and set(df.loc[t.test, "year"]) == {2025}


def test_climatology_falls_back_for_unseen_station():
    tr = pd.DataFrame({"station_id": ["a", "a", "b"], "state": ["K", "K", "K"], "month": [1, 1, 1]})
    m = Climatology().fit(tr, pd.Series([10.0, 20.0, 40.0]))
    te = pd.DataFrame({"station_id": ["a", "z", "z"], "state": ["K", "K", "Q"], "month": [1, 1, 1]})
    assert m.predict(te).tolist() == [15.0, (10 + 20 + 40) / 3, (10 + 20 + 40) / 3]


def test_two_tier_uses_a_where_available_else_b():
    from src.eval.report import derive_tiers

    base = {"split": "temporal", "fold": "f", "model": "random_forest", "y": 10.0}
    rows = [{**base, "station_id": "s", "date_ist": d, "tier": "tier_b", "pred": 1.0} for d in ["d1", "d2"]]
    rows.append({**base, "station_id": "s", "date_ist": "d1", "tier": "tier_a", "pred": 5.0})
    out = derive_tiers(pd.DataFrame(rows))
    two = out[out["tier"] == "two_tier"].set_index("date_ist")["pred"]
    assert two["d1"] == 5.0 and two["d2"] == 1.0
    assert out[out["tier"] == "tier_b_satrows"]["date_ist"].tolist() == ["d1"]


def test_cloud_bin_labels_missing_as_string():
    from src.eval.report import cloud_bin

    p = pd.DataFrame({"cloud_frac_b10": [0.1, 0.9, np.nan, 1.0]})
    out = cloud_bin(p, [0, 0.2, 0.5, 0.8, 1.0001]).tolist()
    assert out == ["0.0-0.2", "0.8-1.0", "missing", "0.8-1.0"]


def test_bootstrap_matches_metrics_and_zero_difference():
    from src.eval.bootstrap import paired_ci

    rng = np.random.default_rng(1)
    df = pd.DataFrame({"block": np.repeat(np.arange(20), 30), "y": rng.normal(30, 10, 600)})
    df["p1"] = df["y"] + rng.normal(0, 5, 600)
    df["p2"] = df["p1"]
    out = paired_ci(df, "block", "y", ["p1", "p2"], n=200, seed=0)
    ref = metrics(df["y"], df["p1"])
    assert np.isclose(out["r2_p1"], ref["r2"]) and np.isclose(out["rmse_p1"], ref["rmse"])
    assert out["r2_p1_lo"] < out["r2_p1"] < out["r2_p1_hi"]
    assert out["d_r2"] == 0 and out["d_r2_lo"] == 0 and out["d_r2_hi"] == 0


def test_sensitivity_mask_recomputes_flags():
    from src.eval.sensitivity import mask_satellite

    df = pd.DataFrame({"aod_clear_b10": [0.3, 0.4], "aod_clear_n_b10": [100, 5],
                       "no2_trop_b10": [1e-5, 2e-5], "no2_trop_n_b10": [200, 200]})
    out = mask_satellite(df, "b10", df["aod_clear_n_b10"] < 10, ["aod_clear"])
    assert out["sat_aod_b10"].tolist() == [1, 0] and out["sat_no2_b10"].tolist() == [1, 1]
    assert np.isnan(out.loc[1, "aod_clear_b10"]) and df.loc[1, "aod_clear_b10"] == 0.4   # input untouched
