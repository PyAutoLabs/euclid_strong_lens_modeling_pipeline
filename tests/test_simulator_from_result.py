"""
``scripts/simulator.py --from-result`` — the resimulation mode.

The mode had no test at all, and it did not work on a single real result. Every light profile
this pipeline fits is *linear* (``al.lp_linear.Sersic``, an MGE ``Basis`` of
``al.lp_linear.Gaussian``): their intensities are solved inside the likelihood rather than
sampled, so they are not model parameters. The tracer the simulator used to rebuild — from
``files/model.json`` plus the maximum-log-likelihood parameter vector — therefore carried no
flux, and every mock it wrote was pure noise. A dark lens is a perfectly valid tracer, so
nothing downstream complained; the only symptom was a mock whose peak SNR was 3.9 where the
real tile's was 81.

The solved intensities were on disk the whole time. PyAutoLens writes ``files/tracer.json`` in
``AnalysisDataset.save_results`` from ``ResultImaging.max_log_likelihood_tracer``, which is
``fit.model_obj_linear_light_profiles_to_light_profiles`` — the maximum-log-likelihood tracer
with the linear profiles already converted to standard ones carrying their solved intensity.

What these tests pin:

- ``tracer_from_result`` returns those intensities, and a tracer that actually makes an image;
- the result directory is still resolved by ``resolve_files_path`` (``--result_hash`` explicit,
  and the newest-converged fallback), so this script and ``tools/diagnose_latent.py`` keep
  agreeing on "the last run";
- the guard raises rather than simulating a dark lens — for a bare linear profile, for a
  linear profile hidden inside an MGE ``Basis`` (the case ``cls_list_from`` does not see), for
  an all-zero-intensity tracer, and for a result with no ``tracer.json`` at all.

No search is run and no dataset is read: each test writes a two-galaxy ``tracer.json`` and the
two files ``resolve_files_path`` tests for. The module is a fraction of a second once
PyAutoLens is imported.
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402

from scripts import simulator  # noqa: E402


SAMPLE = "a_sample"
DATASET = "a_dataset"
UNIQUE_TAG = "sersic_lens_model_baseline"
SEARCH = "vis"
RESULT_HASH = "0123456789abcdef0123456789abcdef"


def write_result(output_path, tracer, result_hash=RESULT_HASH, tracer_json=True):
    """
    The smallest on-disk result ``tracer_from_result`` can read: a ``files/`` directory holding
    ``tracer.json`` plus the two files ``resolve_files_path`` tests for when it is asked for
    "the newest converged result".

    ``samples_summary.json`` and ``model.json`` are placeholders — nothing reads them any more,
    they are only what marks a hash directory as converged.
    """
    files_path = (
        output_path / SAMPLE / DATASET / UNIQUE_TAG / SEARCH / result_hash / "files"
    )
    files_path.mkdir(parents=True)

    (files_path / "samples_summary.json").write_text("{}")
    (files_path / "model.json").write_text("{}")

    if tracer_json:
        (files_path / "tracer.json").write_text(json.dumps(af.to_dict(tracer)))

    return files_path


def args_for(result_hash=RESULT_HASH):
    return Namespace(
        sample=SAMPLE,
        dataset=DATASET,
        unique_tag=UNIQUE_TAG,
        search=SEARCH,
        result_hash=result_hash,
    )


def tracer_from(lens_light, source_light=None):
    if source_light is None:
        source_light = al.lp.Sersic(intensity=3.0, effective_radius=0.2)

    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                bulge=lens_light,
                mass=al.mp.Isothermal(einstein_radius=1.2),
            ),
            al.Galaxy(redshift=1.0, bulge=source_light),
        ]
    )


def test_intensities_are_recovered_and_the_tracer_makes_an_image(tmp_path):
    """
    The defect, directly: the returned lens and source carry the intensities the fit solved for,
    and the image they make is not zero.
    """
    write_result(tmp_path, tracer_from(al.lp.Sersic(intensity=0.5)))

    tracer, files_path = simulator.tracer_from_result(
        args=args_for(), output_path=tmp_path
    )

    assert files_path.name == "files"
    assert files_path.parent.name == RESULT_HASH

    assert tracer.galaxies[0].bulge.intensity == pytest.approx(0.5)
    assert tracer.galaxies[-1].bulge.intensity == pytest.approx(3.0)

    grid = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.1)
    assert tracer.image_2d_from(grid=grid).native.max() > 0.0


def test_no_linear_light_profile_survives(tmp_path):
    """
    The regression assertion the fix exists for: nothing returned by ``tracer_from_result`` may
    still be a linear profile, whose intensity lives in the likelihood rather than on the
    object.
    """
    write_result(tmp_path, tracer_from(al.lp.Sersic(intensity=0.5)))

    tracer, _ = simulator.tracer_from_result(args=args_for(), output_path=tmp_path)

    profiles = [
        profile
        for galaxy in tracer.galaxies
        for profile in simulator.light_profile_list_from(galaxy)
    ]
    assert profiles
    assert not any(
        isinstance(profile, al.lp_linear.LightProfileLinear) for profile in profiles
    )


def test_mge_basis_is_flattened_and_kept(tmp_path):
    """
    An MGE lens light is a ``Basis``; its components are what carry the intensity. The walk has
    to recurse into ``profile_list`` — ``Galaxy.cls_list_from`` does not.
    """
    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp.Gaussian(intensity=1.0, sigma=0.1),
            al.lp.Gaussian(intensity=2.0, sigma=0.4),
        ]
    )
    write_result(tmp_path, tracer_from(basis))

    tracer, _ = simulator.tracer_from_result(args=args_for(), output_path=tmp_path)

    intensities = [
        profile.intensity
        for profile in simulator.light_profile_list_from(tracer.galaxies[0])
    ]
    assert intensities == pytest.approx([1.0, 2.0])


def test_a_linear_lens_light_raises(tmp_path):
    """
    The failure this bug shipped as, now loud: a ``tracer.json`` that still holds linear
    profiles must stop the run rather than simulate a dark lens.
    """
    write_result(tmp_path, tracer_from(al.lp_linear.Sersic()))

    with pytest.raises(SystemExit, match="linear light profiles"):
        simulator.tracer_from_result(args=args_for(), output_path=tmp_path)


def test_a_linear_profile_inside_an_mge_basis_raises(tmp_path):
    """
    The same, hidden one level down — the MGE case, which is what
    ``initial_lens_model/vis_lp`` results are made of.
    """
    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Gaussian(sigma=0.1),
            al.lp_linear.Gaussian(sigma=0.4),
        ]
    )
    write_result(tmp_path, tracer_from(basis))

    with pytest.raises(SystemExit, match="linear light profiles"):
        simulator.tracer_from_result(args=args_for(), output_path=tmp_path)


def test_an_all_dark_tracer_raises(tmp_path):
    """
    Standard profiles, but every intensity zero: still a mock of pure noise, so still a stop.
    """
    write_result(
        tmp_path,
        tracer_from(
            al.lp.Sersic(intensity=0.0),
            source_light=al.lp.Sersic(intensity=0.0),
        ),
    )

    with pytest.raises(SystemExit, match="zero intensity"):
        simulator.tracer_from_result(args=args_for(), output_path=tmp_path)


def test_a_result_without_tracer_json_raises(tmp_path):
    """
    A result predating ``save_results`` writing ``tracer.json`` cannot be resimulated. Say so,
    naming the directory, rather than failing later on a missing attribute.
    """
    write_result(tmp_path, tracer_from(al.lp.Sersic(intensity=0.5)), tracer_json=False)

    with pytest.raises(SystemExit, match="no tracer.json"):
        simulator.tracer_from_result(args=args_for(), output_path=tmp_path)


def test_the_newest_converged_result_is_used_when_no_hash_is_given(tmp_path):
    """
    ``--result_hash`` omitted still means "the newest converged result", resolved by
    ``tools/diagnose_latent.py::resolve_files_path`` — the fix must not have changed which
    directory is picked.
    """
    old = write_result(
        tmp_path, tracer_from(al.lp.Sersic(intensity=0.5)), result_hash="a" * 32
    )
    new = write_result(
        tmp_path, tracer_from(al.lp.Sersic(intensity=0.9)), result_hash="b" * 32
    )
    import os
    import time

    os.utime(old, (time.time() - 1000, time.time() - 1000))

    tracer, files_path = simulator.tracer_from_result(
        args=args_for(result_hash=None), output_path=tmp_path
    )

    assert files_path == new
    assert tracer.galaxies[0].bulge.intensity == pytest.approx(0.9)
