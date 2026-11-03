import pandas as pd

from src.ingest.cpcb_stations import apply_manual_coordinates, match_to_openaq, normalise_name


def test_normalise_name_ignores_case_and_punctuation():
    assert normalise_name("SIPCOT Phase-1, Hosur - TNPCB") == normalise_name("sipcot phase 1 hosur  tnpcb")


def test_match_accepts_close_duplicates_and_flags_far_ones():
    openaq = pd.DataFrame({
        "location_id": [1, 2, 3, 4],
        "name": ["Site A, X - PCB", "Site A, X - PCB", "Site B, Y - PCB", "Site B, Y - PCB"],
        "latitude": [10.0, 10.0, 12.0, 13.0],     # B's two entries are ~111 km apart
        "longitude": [76.0, 76.0, 77.0, 77.0],
    })
    cpcb = pd.DataFrame({"station": ["Site A, X - PCB", "Site B, Y - PCB", "Site C, Z - PCB"]})
    out = match_to_openaq(cpcb, openaq)
    assert out["coord_source"].tolist() == ["openaq_name_match", "coord_conflict", "unmatched"]
    assert out.loc[0, "openaq_location_ids"] == "1;2"
    assert out.loc[0, "latitude"] == 10.0
    assert pd.isna(out.loc[1, "latitude"])


def test_manual_coordinates_fill_only_missing():
    stations = pd.DataFrame({"station": ["Site A, X - PCB", "Site B, Y - PCB"],
                             "latitude": [10.0, float("nan")], "longitude": [76.0, float("nan")],
                             "coord_source": ["openaq_name_match", "coord_conflict"]})
    manual = pd.DataFrame({"station": ["site a x pcb", "Site B, Y - PCB"], "latitude": [99.0, 12.5],
                           "longitude": [99.0, 77.5], "source": ["dash", "dash"]})
    out = apply_manual_coordinates(stations, manual)
    assert out.loc[0, "latitude"] == 10.0          # existing match not overridden
    assert out.loc[1, ["latitude", "longitude"]].tolist() == [12.5, 77.5]
    assert out.loc[1, "coord_source"] == "manual:dash"
