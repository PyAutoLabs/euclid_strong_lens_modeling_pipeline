"""project_paths.dataset_root(): environment, then hpc/sync.conf, then <project>/dataset."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import project_paths  # noqa: E402


def _fresh(monkeypatch, tmp_path, conf=None, env=None):
    monkeypatch.setattr(project_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(project_paths, "SYNC_CONF", tmp_path / "hpc" / "sync.conf")
    if conf is not None:
        (tmp_path / "hpc").mkdir()
        (tmp_path / "hpc" / "sync.conf").write_text(conf)
    if env is None:
        monkeypatch.delenv("DATASET_ROOT", raising=False)
    else:
        monkeypatch.setenv("DATASET_ROOT", env)


def test_default_is_the_project_dataset_folder(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    assert project_paths.dataset_root() == tmp_path / "dataset"


def test_sync_conf_key_wins_over_the_default(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, conf="HPC_HOST=x\nDATASET_ROOT=/data/shared  # laptop\n")
    assert project_paths.dataset_root() == Path("/data/shared")


def test_a_relative_value_is_from_the_project_root(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, conf='DATASET_ROOT="../shared"\n')
    assert project_paths.dataset_root() == tmp_path / "../shared"


def test_the_environment_wins_over_sync_conf(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, conf="DATASET_ROOT=/data/shared\n", env="/data/env")
    assert project_paths.dataset_root() == Path("/data/env")
