"""Offline tests for the DR1 campaign: ``hpc/campaign/dr1.yaml`` and what it runs.

Nothing is submitted. The campaign is loaded and validated with ``autofit.campaign``,
dry-run and "submitted" to its in-memory ``FakeScheduler`` on a three-lens fixture
manifest, and its digest is collected. The stage entry (``hpc/batch_cpu/stage``) and
the seed scripts run under bash against a throwaway project.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DR1 = ROOT / "hpc" / "campaign" / "dr1.yaml"
STAGE = ROOT / "hpc" / "batch_cpu" / "stage"
SED_SEED = ROOT / "hpc" / "batch_cpu" / "sed_seed.sh"
VIS_PIX_SEED = ROOT / "hpc" / "batch_cpu" / "vis_pix_seed.sh"
FIXTURE_MANIFEST = ROOT / "tests" / "fixtures" / "hpc_campaign" / "dr1_manifest.txt"
BUILDER = ROOT / "hpc" / "tools" / "build_campaign_manifest.py"


def _campaign_module():
    return pytest.importorskip("autofit.campaign.spec")


@pytest.fixture
def campaign(tmp_path):
    """dr1.yaml with its project root and manifest moved onto the three-lens fixture."""
    spec = _campaign_module()
    raw = yaml.safe_load(DR1.read_text())
    (tmp_path / "hpc" / "campaign").mkdir(parents=True)
    shutil.copy(FIXTURE_MANIFEST, tmp_path / "hpc" / "campaign" / "dr1_manifest.txt")
    raw["project_root"] = str(tmp_path)
    raw["cache_root"] = str(tmp_path / "cache")
    path = tmp_path / "hpc" / "campaign" / "dr1.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return spec.Campaign.load(path)


def test_dr1_campaign_validates_against_the_ral_site():
    spec = _campaign_module()
    campaign = spec.Campaign.load(DR1)

    assert campaign.name == "euclid_dr1"
    assert campaign.order() == ["vis_lp", "vis_pix", "sed"]
    assert campaign.site is not None and campaign.site.name == "ral"


def test_every_stage_is_on_ral_and_passes_the_partition_lint():
    spec = _campaign_module()
    from autofit.campaign.lint import check_partitions

    campaign = spec.Campaign.load(DR1)
    for stage in campaign.stages.values():
        resources = stage.resources
        assert resources.partition == "ral"
        assert resources.gres is None
        verdict = check_partitions(
            resources.partition,
            cpus=resources.cpus,
            gres=resources.gres,
            throttle=resources.throttle,
            n_tasks=1000,
            pending_gpu_jobs=None,
            site=campaign.site,
        )
        assert verdict.ok, verdict.reasons
    # No stage block mentions a GPU partition, even in a comment.
    assert "gpu" not in DR1.read_text().split("\nstages:\n")[1]


def test_the_lint_refuses_the_campaign_on_a_gpu_partition(tmp_path):
    spec = _campaign_module()
    raw = yaml.safe_load(DR1.read_text())
    raw["stages"]["vis_pix"]["resources"]["partition"] = "ral,gpu"
    path = tmp_path / "dr1.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(spec.CampaignException, match="vis_pix"):
        spec.Campaign.load(path)


def test_stages_chain_per_lens():
    spec = _campaign_module()
    campaign = spec.Campaign.load(DR1)

    assert campaign.stages["vis_lp"].after == {}
    assert campaign.stages["vis_pix"].after == {"vis_lp": "aftercorr"}
    assert campaign.stages["sed"].after == {"vis_pix": "aftercorr"}


def test_stage_resources_and_environment():
    spec = _campaign_module()
    campaign = spec.Campaign.load(DR1)
    stages = campaign.stages

    assert {name: (s.resources.cpus, s.resources.mem_mb) for name, s in stages.items()} == {
        "vis_lp": (4, 12288),
        "vis_pix": (8, 16384),
        "sed": (8, 16384),
    }
    for name, stage in stages.items():
        assert stage.script == "hpc/batch_cpu/stage"
        assert stage.env["PYAUTO_CONFIG_OVERLAY"] == "hpc/config"
        cores = f"--number_of_cores={stage.resources.cpus}"
        assert cores in stage.env["STAGE_ARGS"].split(), name
        assert (ROOT / "scripts" / stage.env["STAGE_SCRIPT"]).is_file()
    assert stages["vis_pix"].env["BLAS_THREADS"] == "1"
    assert "--use_cpu" in stages["vis_pix"].env["STAGE_ARGS"]
    # vis_lp and the SED chain run JAX on the CPU backend: --use_cpu would turn it off.
    assert "--use_cpu" not in stages["vis_lp"].env["STAGE_ARGS"]
    assert "--use_cpu" not in stages["sed"].env["STAGE_ARGS"]
    assert stages["sed"].env["PYAUTO_OUTPUT_DIR"] == "output_sed"
    for name in ("vis_pix", "sed"):
        setup = ROOT / stages[name].env["STAGE_SETUP"]
        assert setup.is_file() and os.access(setup, os.X_OK)
    assert os.access(STAGE, os.X_OK)


def test_dry_run_renders_safe_batch_scripts(campaign):
    from autofit.campaign.submit import submit

    entries = submit(campaign, None, dry_run=True)

    assert [entry["stage"] for entry in entries] == ["vis_lp", "vis_pix", "sed"]
    for entry in entries:
        script = entry["script"]
        assert entry["lint"] == "ok"
        assert entry["indices"] == "0-2"
        assert "#SBATCH --partition=ral\n" in script
        assert "#SBATCH --signal=B:USR1@600" in script
        assert "set -euo pipefail" in script
        assert "export PYTHONUNBUFFERED=1" in script
        assert "gpu" not in script.lower()
        assert "hpc/batch_cpu/stage &" in script
        assert f"cd {campaign.project_root}" in script
    vis_lp, vis_pix, sed = (entry["script"] for entry in entries)
    assert "--dependency" not in vis_lp
    assert "#SBATCH --dependency=aftercorr:<vis_lp@0>" in vis_pix
    assert "#SBATCH --dependency=aftercorr:<vis_pix@0>" in sed
    assert "export BLAS_THREADS=1" in vis_pix
    assert "export STAGE_ARGS='--stage=vis_lp --number_of_cores=4'" in vis_lp
    assert "export PYAUTO_OUTPUT_DIR=output_sed" in sed


def test_fake_scheduler_submit_chains_each_stage_on_the_one_before(campaign):
    from autofit.campaign.scheduler.fake import FakeScheduler
    from autofit.campaign.submit import read_ledger, submit

    scheduler = FakeScheduler(site=campaign.site)
    entries = submit(campaign, scheduler)

    jobs = {entry["stage"]: entry["job_id"] for entry in entries}
    requests = scheduler.requests
    assert requests[jobs["vis_lp"]].after == {}
    assert requests[jobs["vis_pix"]].after == {jobs["vis_lp"]: "aftercorr"}
    assert requests[jobs["sed"]].after == {jobs["vis_pix"]: "aftercorr"}
    for request in requests.values():
        assert request.resources.partition == "ral"
        assert request.indices == [0, 1, 2]
    assert [entry["stage"] for entry in read_ledger(campaign)] == ["vis_lp", "vis_pix", "sed"]


def test_status_digest_fits_in_two_kilobytes(campaign):
    from autofit.campaign.collector import collect
    from autofit.campaign.digest import digest_markdown
    from autofit.campaign.scheduler.fake import FakeScheduler
    from autofit.campaign.submit import submit

    scheduler = FakeScheduler(
        outcomes={("vis_lp", 1): {"scheduler_state": "TIMEOUT"}}, site=campaign.site
    )
    submit(campaign, scheduler)
    scheduler.run()

    digest = collect(campaign, scheduler)
    markdown = digest_markdown(digest)

    assert len(markdown.encode()) <= 2048
    assert (campaign.directory / "digest.md").stat().st_size <= 2048
    states = {(o["stage"], o["index"]): o["state"] for o in map(json.loads, (campaign.directory / "tasks.jsonl").read_text().splitlines())}
    assert states[("vis_lp", 1)] != "completed"
    # vis_pix and sed of lens 1 never ran: their aftercorr dependency was not met.
    assert states[("vis_pix", 0)] == "completed"
    assert states[("vis_pix", 1)] != "completed"


# ---------------------------------------------------------------------------
# The stage entry and the seed scripts
# ---------------------------------------------------------------------------


@pytest.fixture
def stage_project(tmp_path):
    """A project with a two-lens manifest and a stand-in pipeline script."""
    project = tmp_path / "proj"
    (project / "hpc" / "batch_cpu").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "hpc" / "manifest.txt").write_text("dr1_sep1_rest lens_a\ndr1_sep1_rest lens_b\n")
    (project / "scripts" / "fake.py").write_text(
        "import json, os, sys\n"
        "json.dump({'args': sys.argv[1:], 'env': {k: os.environ.get(k) for k in\n"
        "    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',\n"
        "     'NUMBA_NUM_THREADS', 'NPROC', 'JAX_PLATFORMS')}},\n"
        "    open(os.environ['FAKE_OUT'], 'w'))\n"
    )
    for script in (SED_SEED, VIS_PIX_SEED):
        shutil.copy(script, project / "hpc" / "batch_cpu" / script.name)
    # The stand-in runs without site-packages (-S): a local venv's .pth files may pin
    # thread variables at start-up, which would hide what the stage exported.
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nexec python3 -S "$@"\n')
    python.chmod(0o755)

    def run(index, **extra):
        env = {
            **os.environ,
            "PROJECT_PATH": str(project),
            "STAGE": "vis_pix",
            "STAGE_SCRIPT": "fake.py",
            "MANIFEST": "hpc/manifest.txt",
            "CAMPAIGN": "demo",
            "SLURM_ARRAY_TASK_ID": str(index),
            "SLURM_ARRAY_JOB_ID": "777",
            "SLURM_CPUS_PER_TASK": "8",
            "FAKE_OUT": str(tmp_path / "fake.json"),
            "PYTHON": str(python),
            **extra,
        }
        for key in ("BLAS_THREADS", "STAGE_SETUP", "STAGE_ARGS", "PYAUTO_OUTPUT_DIR"):
            if key not in extra:
                env.pop(key, None)
        return subprocess.run(
            ["bash", str(STAGE)], cwd=project, env=env, capture_output=True, text=True, timeout=60
        )

    def fake_output():
        return json.loads((tmp_path / "fake.json").read_text())

    return project, run, fake_output


def _record(project, stage, index):
    return project / "output" / ".campaign_tasks" / "demo" / stage / f"{index}.status.json"


def _vis_lp_zip(project, dataset, root="output"):
    folder = project / root / "dr1_sep1_rest" / dataset / "initial_lens_model" / "vis_lp"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "abc123.zip").write_bytes(b"zip")
    return folder / "abc123.zip"


def test_stage_threads_default_to_the_allocated_cpus(stage_project):
    project, run, fake_output = stage_project

    process = run(0)

    assert process.returncode == 0, process.stderr
    out = fake_output()
    assert out["args"] == ["--sample=dr1_sep1_rest", "--dataset=lens_a", "--use_cpu", "--number_of_cores=8"]
    assert out["env"] == {
        "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "NUMBA_NUM_THREADS": "8", "NPROC": "8", "JAX_PLATFORMS": "cpu",
    }


def test_stage_blas_threads_pins_blas_and_numba_but_not_nproc(stage_project):
    project, run, fake_output = stage_project

    process = run(1, BLAS_THREADS="1", STAGE_ARGS="--stage=vis_pix --use_cpu --number_of_cores=8")

    assert process.returncode == 0, process.stderr
    out = fake_output()
    assert out["args"] == [
        "--sample=dr1_sep1_rest", "--dataset=lens_b", "--stage=vis_pix", "--use_cpu", "--number_of_cores=8",
    ]
    env = out["env"]
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["NUMBA_NUM_THREADS"] == "1"
    assert env["NPROC"] == "8"


def test_stage_setup_runs_with_the_manifest_fields_before_the_fit(stage_project, tmp_path):
    project, run, fake_output = stage_project
    (project / "hpc" / "setup.sh").write_text(
        f'#!/bin/bash\necho "$sample $dataset $*" > "{tmp_path}/setup.txt"\n'
    )
    (project / "hpc" / "setup.sh").chmod(0o755)

    process = run(1, STAGE_SETUP="hpc/setup.sh")

    assert process.returncode == 0, process.stderr
    assert (tmp_path / "setup.txt").read_text().strip() == (
        "dr1_sep1_rest lens_b --sample=dr1_sep1_rest --dataset=lens_b"
    )
    assert fake_output()["args"][:2] == ["--sample=dr1_sep1_rest", "--dataset=lens_b"]
    assert not _record(project, "vis_pix", 1).exists()


@pytest.mark.parametrize("setup_exit, failure_class", [(3, "missing_seed"), (1, "bad_input")])
def test_a_failed_stage_setup_fails_the_task_without_running_the_fit(
    stage_project, tmp_path, setup_exit, failure_class
):
    project, run, _ = stage_project
    (project / "hpc" / "setup.sh").write_text(f"#!/bin/bash\nexit {setup_exit}\n")
    (project / "hpc" / "setup.sh").chmod(0o755)

    process = run(0, STAGE_SETUP="hpc/setup.sh")

    assert process.returncode == setup_exit
    assert not (tmp_path / "fake.json").exists()
    record = json.loads(_record(project, "vis_pix", 0).read_text())
    assert record["state"] == "failed"
    assert record["failure_class"] == failure_class
    assert record["exit"] == setup_exit


def test_vis_pix_without_its_vis_lp_seed_is_missing_seed(stage_project, tmp_path):
    project, run, _ = stage_project

    process = run(0, STAGE_SETUP="hpc/batch_cpu/vis_pix_seed.sh")

    assert process.returncode == 3
    assert "found 0" in process.stderr
    assert not (tmp_path / "fake.json").exists()
    assert json.loads(_record(project, "vis_pix", 0).read_text())["failure_class"] == "missing_seed"

    _vis_lp_zip(project, "lens_a")
    assert run(0, STAGE_SETUP="hpc/batch_cpu/vis_pix_seed.sh").returncode == 0
    assert not _record(project, "vis_pix", 0).exists()


def _sed_seed(project, dataset):
    return subprocess.run(
        ["bash", str(project / "hpc" / "batch_cpu" / "sed_seed.sh")],
        cwd=project,
        env={**os.environ, "sample": "dr1_sep1_rest", "dataset": dataset, "PYAUTO_OUTPUT_DIR": "output_sed"},
        capture_output=True, text=True, timeout=60,
    )


def test_sed_seed_with_no_vis_lp_zip_exits_3(stage_project):
    project, _, _ = stage_project

    process = _sed_seed(project, "lens_a")

    assert process.returncode == 3
    assert "found 0" in process.stderr
    assert not (project / "output_sed").exists()


def test_sed_seed_with_two_vis_lp_zips_exits_3(stage_project):
    project, _, _ = stage_project
    seed = _vis_lp_zip(project, "lens_a")
    (seed.parent / "def456.zip").write_bytes(b"zip")

    assert _sed_seed(project, "lens_a").returncode == 3


def test_sed_seed_copies_the_one_vis_lp_zip(stage_project):
    project, _, _ = stage_project
    _vis_lp_zip(project, "lens_a")

    process = _sed_seed(project, "lens_a")

    assert process.returncode == 0, process.stderr
    copied = project / "output_sed" / "dr1_sep1_rest" / "lens_a" / "initial_lens_model" / "vis_lp" / "abc123.zip"
    assert copied.read_bytes() == b"zip"


def test_sed_stage_copies_its_seed_then_runs(stage_project):
    project, run, fake_output = stage_project
    _vis_lp_zip(project, "lens_b")

    process = run(
        1, STAGE="sed", STAGE_SETUP="hpc/batch_cpu/sed_seed.sh",
        PYAUTO_OUTPUT_DIR="output_sed", STAGE_ARGS="--number_of_cores=8",
    )

    assert process.returncode == 0, process.stderr
    assert (project / "output_sed" / "dr1_sep1_rest" / "lens_b" / "initial_lens_model" / "vis_lp" / "abc123.zip").exists()
    assert fake_output()["args"] == ["--sample=dr1_sep1_rest", "--dataset=lens_b", "--number_of_cores=8"]


# ---------------------------------------------------------------------------
# The cluster config overlay and the manifest builder
# ---------------------------------------------------------------------------


def test_cluster_overlay_only_turns_off_the_numba_disk_cache():
    overlay = yaml.safe_load((ROOT / "hpc" / "config" / "general.yaml").read_text())
    assert overlay == {"numba": {"cache": False}}


def test_manifest_builder_lists_lenses_with_their_fits_sorted(tmp_path):
    root = tmp_path / "dataset"
    for sample, name in [("s1", "b"), ("s1", "a"), ("s2", "c"), ("s1", "excluded")]:
        (root / sample / name).mkdir(parents=True)
        (root / sample / name / f"{name}.fits").write_bytes(b"")
    (root / "s1" / "no_fits").mkdir()
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("# done already\ns1 excluded\n")
    output = tmp_path / "manifest.txt"

    process = subprocess.run(
        ["python3", str(BUILDER), "--sample", "s1", "--sample", "s2", "--dataset-root", str(root),
         "--exclude-file", str(exclude), "--output", str(output)],
        capture_output=True, text=True, timeout=60,
    )

    assert process.returncode == 0, process.stderr
    assert output.read_text() == "s1 a\ns1 b\ns2 c\n"


def test_fixture_manifest_is_three_real_tiles():
    lines = FIXTURE_MANIFEST.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        sample, tile = line.split()
        assert sample == "dr1_sep1_rest"
        assert (ROOT / "tests" / "fixtures" / "positions_gate" / tile).is_dir()
