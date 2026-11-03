import csv

from src.utils.config import config_hash
from src.utils.manifest import record_file
from src.utils.run_log import log_run


def test_config_hash_ignores_key_order_but_not_values():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_manifest_keeps_one_row_per_file(tmp_path):
    data = tmp_path / "x.csv"
    data.write_text("a\n1\n")
    manifest = tmp_path / "manifest.csv"
    record_file(data, "test", "2021", rows=1, manifest_path=manifest)
    data.write_text("a\n1\n2\n")
    record_file(data, "test", "2021", rows=2, manifest_path=manifest)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["rows"] == "2"
    assert len(rows[0]["sha256"]) == 64


def test_run_log_appends(tmp_path):
    path = tmp_path / "runs.csv"
    log_run("step1", "abc", {"n": 1}, path=path)
    log_run("step2", "abc", path=path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [r["step"] for r in rows] == ["step1", "step2"]
