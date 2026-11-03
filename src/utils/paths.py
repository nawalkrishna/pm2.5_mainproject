"""Project-relative paths, so scripts work from any current directory."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM: Path = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
REPORTS_FIGURES: Path = PROJECT_ROOT / "reports" / "figures"
REPORTS_TABLES: Path = PROJECT_ROOT / "reports" / "tables"
LOG_DIR: Path = PROJECT_ROOT / "logs"


def resolve(path_str: str) -> Path:
    """Turn a project-relative path from a config file into an absolute path."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path
