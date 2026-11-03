# Satellite-Based Ground-Level PM2.5 Estimation Using Multi-Source Data

MTech dissertation, IIIT Kottayam. Phase 1: data audit, acquisition, cleaning,
alignment, features, baselines and evaluation framework for South India,
with Kerala as the monsoon focus region.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env    # then put your OpenAQ API key in .env
```

## Running

```powershell
python -m src.run stage1     # boundaries + OpenAQ download + station audit
python -m pytest             # tests
```

Downloads are resumable: files that already exist in `data/raw` are not fetched again.

## Layout

| Folder | Contents |
|---|---|
| `configs/` | YAML settings: study area, audit thresholds, sources, features |
| `data/raw/` | Downloaded files, never edited by hand (not in git) |
| `data/interim/`, `data/processed/` | Script outputs (not in git) |
| `src/ingest/` | Download and standardise sources |
| `src/clean/`, `src/align/`, `src/features/` | Cleaning, spatial/temporal alignment, features |
| `src/audit/` | Stage 1 station audit |
| `src/models/`, `src/eval/` | Models and the shared evaluation framework |
| `src/utils/` | Paths, config, logging, run log, data manifest |
| `reports/` | `runs.csv` run log, `tables/` (incl. `data_manifest.csv`), `figures/` |
| `docs/` | Decision log, data dictionary, station audit, Phase 1 report |
| `tests/` | pytest tests |

Every scientific choice is recorded in `docs/decision_log.md`.
