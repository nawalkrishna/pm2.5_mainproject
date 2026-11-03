import pandas as pd

from src.audit.cpcb_audit import link_sites_to_stations


def test_link_ignores_dropped_separators():
    files = pd.DataFrame({"cpcb_site": ["site_1", "site_2"],
                          "slug": ["rvcemailasandra_bengaluru_kspcb", "sipcot_phase1_hosur_tnpcb"]})
    stations = pd.DataFrame({"state": ["Karnataka", "Tamil Nadu"], "city": ["Bengaluru", "Hosur"],
                             "station": ["RVCE-Mailasandra, Bengaluru - KSPCB", "SIPCOT Phase-1, Hosur - TNPCB"]})
    out = link_sites_to_stations(files, stations)
    assert out["station"].notna().all()
