import numpy as np
import pandas as pd

from src.clean.pm25_qc import apply_qc, cleaning_log

CFG = {"invalid": {"negative": True, "zero": True, "exact_values": [1000]}, "flat_run_hours": 4,
       "spike": {"factor": 4.0, "min_value": 100.0, "window_hours": 25, "min_periods": 5, "max_isolated_run": 2}}


def make(values):
    t = pd.date_range("2021-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({"station_id": "s1", "timestamp_utc": t, "pm25": values})


def test_rules_in_order_and_log_reconciles():
    rng = np.random.default_rng(1)
    v = list(20 + rng.normal(0, 2, 40))
    v[3], v[5], v[7] = -1.0, 0.0, 1000.0
    v[10:15] = [33.0] * 5                         # flat run of 5 >= 4
    v[25] = 400.0                                 # isolated spike
    v[30:34] = [300.0, 310.0, 320.0, 330.0]       # 4-hour episode: kept
    q = apply_qc(make(v), CFG)
    flags = q["qc_flag"].tolist()
    assert flags[3] == "invalid_negative" and flags[5] == "invalid_zero" and flags[7] == "invalid_exact"
    assert set(flags[10:15]) == {"flat_run"}
    assert flags[25] == "spike_isolated"
    assert set(flags[30:34]) == {"ok"}
    log = cleaning_log(q)
    assert log["rows_after"].iloc[-1] == (q["qc_flag"] == "ok").sum()
    assert (log["rows_before"] - log["rows_removed"] == log["rows_after"]).all()
