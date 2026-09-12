"""
The four Sersic-stage variants composed by ``scripts/sersic_lens_model.py``.

``sersic_model_from`` and ``unique_tag_from`` are the whole of the variant
behaviour, split out of ``fit_sersic`` so they can be built without a dataset and
without a search. What these tests pin is the part that is easy to break silently:

- that **no variant** is byte-for-byte the fit that existed before variants did —
  the same twelve-parameter model, the same ``sersic_lens_model`` tag;
- that ``wide_n`` really widens the lens Sersic index past the config edge, and
  really leaves the source's alone. The config prior carries ``limits: 0.8-5.0``
  (``config/priors/light/linear/sersic.yaml``), so an assigned prior being quietly
  clipped back to them is the failure mode worth a test;
- that ``sersic_point`` costs exactly four parameters, and that its Gaussians span
  the intended sigma range;
- that the CLI accepts the four names and rejects anything else.

No search is run and JAX is never imported; this module is a fraction of a second.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import util  # noqa: E402
import sersic_lens_model  # noqa: E402


# The unmodified model: a six-parameter linear Sersic for each galaxy, with the
# mass and shear passed as instances.
BASELINE_PRIOR_COUNT = 12

# `sersic_point` adds a Basis of five linear Gaussians sharing one centre (2) and
# one pair of ell_comps (2).
POINT_PRIOR_COUNT = 16

# The config prior for a linear Sersic's index (config/priors/light/linear/sersic.yaml).
CONFIG_SERSIC_INDEX_LIMITS = (0.8, 5.0)

PIXEL_SCALES = 0.1


@pytest.fixture(scope="module", autouse=True)
def push_config():
    """
    Point the library at this repository's ``config/``, which is where the Sersic
    prior defaults the models are built from come from.
    """
    from autolens import conf

    conf.instance.push(
        new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
    )


def model_from(variant):
    """
    The variant's model, built on stub centre priors and stub mass/shear instances.

    The real centres come from ``vis_result.model_centred``; nothing under test
    depends on their values, only on there being two priors per galaxy.
    """
    import autofit as af
    import autolens as al

    def centre():
        # A fresh pair per galaxy: the lens's and the source's centres are four
        # distinct priors in the real model, and sharing them here would undercount.
        return (
            af.GaussianPrior(mean=0.0, sigma=0.1),
            af.GaussianPrior(mean=0.0, sigma=0.1),
        )

    return sersic_lens_model.sersic_model_from(
        lens_centre=centre(),
        source_centre=centre(),
        mass=al.mp.Isothermal(),
        shear=al.mp.ExternalShear(),
        variant=variant,
        pixel_scales=PIXEL_SCALES,
        point_centre=(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# The unmodified model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", [None, "baseline", "central_noise"])
def test_unmodified_variants_are_the_twelve_parameter_model(variant):
    """
    ``None``, ``baseline`` and ``central_noise`` all compose the same model.
    ``baseline`` is the re-run of it, and ``central_noise`` changes the noise map
    rather than the model, so only ``wide_n`` and ``sersic_point`` touch this.
    """
    model = model_from(variant)

    assert model.prior_count == BASELINE_PRIOR_COUNT

    for galaxy in (model.galaxies.lens, model.galaxies.source):
        sersic_index = galaxy.bulge.sersic_index
        assert (
            sersic_index.lower_limit,
            sersic_index.upper_limit,
        ) == CONFIG_SERSIC_INDEX_LIMITS

    assert not hasattr(model.galaxies.lens, "point")


def test_unique_tag_without_a_variant_is_unchanged():
    """
    The contract the whole feature rests on: an existing run's output path is not
    moved by the existence of variants.
    """
    assert sersic_lens_model.unique_tag_from(None) == "sersic_lens_model"


@pytest.mark.parametrize("variant", util.VARIANTS)
def test_unique_tag_of_a_variant_sits_beside_it(variant):
    assert sersic_lens_model.unique_tag_from(variant) == f"sersic_lens_model_{variant}"


def test_unique_tag_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        sersic_lens_model.unique_tag_from("nope")


def test_sersic_model_from_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        model_from("nope")


# ---------------------------------------------------------------------------
# wide_n
# ---------------------------------------------------------------------------


def test_wide_n_widens_the_lens_index_and_leaves_the_source_alone():
    """
    The lens index prior must really report (0.5, 10.0). The config entry for this
    profile declares ``limits: 0.8-5.0``, and a prior assignment that got clipped
    back to them would leave the variant testing nothing while looking fine.
    """
    model = model_from("wide_n")

    lens_index = model.galaxies.lens.bulge.sersic_index

    assert (lens_index.lower_limit, lens_index.upper_limit) == (
        sersic_lens_model.WIDE_N_LOWER_LIMIT,
        sersic_lens_model.WIDE_N_UPPER_LIMIT,
    )

    source_index = model.galaxies.source.bulge.sersic_index

    assert (
        source_index.lower_limit,
        source_index.upper_limit,
    ) == CONFIG_SERSIC_INDEX_LIMITS

    # Widening a prior must not add or remove a parameter.
    assert model.prior_count == BASELINE_PRIOR_COUNT


# ---------------------------------------------------------------------------
# sersic_point
# ---------------------------------------------------------------------------


def test_sersic_point_adds_a_five_gaussian_nucleus():
    model = model_from("sersic_point")

    assert model.prior_count == POINT_PRIOR_COUNT

    profile_list = model.galaxies.lens.point.profile_list

    assert len(profile_list) == sersic_lens_model.POINT_TOTAL_GAUSSIANS

    sigma_list = [float(gaussian.sigma) for gaussian in profile_list]

    # Log-spaced from `sigma_min` to twice the pixel scale, fixed (not free).
    assert sigma_list == sorted(sigma_list)
    assert sigma_list[0] == pytest.approx(sersic_lens_model.POINT_SIGMA_MIN)
    assert sigma_list[-1] == pytest.approx(2.0 * PIXEL_SCALES)

    # The Sersic itself is untouched by the nucleus.
    lens_index = model.galaxies.lens.bulge.sersic_index

    assert (
        lens_index.lower_limit,
        lens_index.upper_limit,
    ) == CONFIG_SERSIC_INDEX_LIMITS


def test_sersic_point_needs_a_pixel_scale_and_a_centre():
    """
    The two arguments the nucleus is sized and placed with are not optional for
    this variant; a default would put the Gaussians in the wrong place silently.
    """
    import autofit as af
    import autolens as al

    centre = (
        af.GaussianPrior(mean=0.0, sigma=0.1),
        af.GaussianPrior(mean=0.0, sigma=0.1),
    )

    with pytest.raises(ValueError, match="pixel_scales"):
        sersic_lens_model.sersic_model_from(
            lens_centre=centre,
            source_centre=centre,
            mass=al.mp.Isothermal(),
            shear=al.mp.ExternalShear(),
            variant="sersic_point",
        )


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", util.VARIANTS)
def test_parse_fit_args_round_trips_a_variant(monkeypatch, variant):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sersic_lens_model.py", "--dataset=a_lens", f"--variant={variant}"],
    )

    parsed = util.parse_fit_args(with_variant=True)

    assert len(parsed) == 7
    assert parsed[-1] == variant


def test_parse_fit_args_variant_defaults_to_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sersic_lens_model.py", "--dataset=a_lens"])

    assert util.parse_fit_args(with_variant=True)[-1] is None


def test_parse_fit_args_without_with_variant_is_the_six_tuple(monkeypatch):
    """
    Every other script keeps the tuple it already unpacks.
    """
    monkeypatch.setattr(sys, "argv", ["initial_lens_model.py", "--dataset=a_lens"])

    assert len(util.parse_fit_args()) == 6


def test_parse_fit_args_rejects_an_unknown_variant(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["sersic_lens_model.py", "--dataset=a_lens", "--variant=nope"]
    )

    with pytest.raises(SystemExit):
        util.parse_fit_args(with_variant=True)
