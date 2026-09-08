"""
The ``vis_lp`` model composed by ``scripts/initial_lens_model.py``.

``vis_lp_model_from`` is the whole of that composition, split out of ``fit`` so it can
be built without a dataset and without a search. What these tests pin is the part of it
that is easy to break silently:

- the parameter count, which the search's ``n_live`` is sized against;
- the two ell_comps boxes, +/-0.5 for the lens light and +/-0.7 for the source, which
  used to be imposed by reassigning fresh priors over the returned model and are now
  ``ell_comps_limit`` arguments;
- the single ordering assertion ``order_bases=True`` attaches to the lens bulge, and
  that it actually accepts the ordered labelling and rejects the swapped one.

The last is the point of the change (issue #54, ``docs/mge_label_degeneracy.md``): the
two lens bases are exchangeable, so an unordered fit reports the same physical solution
under either labelling. The assertion has to survive being placed inside the enclosing
``af.Collection``, which is why it is looked for with ``gathered_assertions`` on the top
level model rather than on the ``Basis`` it is attached to.

No search is run and JAX is never imported; this module is a fraction of a second.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import initial_lens_model  # noqa: E402


MASK_RADIUS = 3.5
DATASET_CENTRE = (0.0, 0.0)
REDSHIFT_LENS = 0.5
REDSHIFT_SOURCE = 1.0

LENS_ELL_COMPS_LIMIT = 0.5
SOURCE_ELL_COMPS_LIMIT = 0.7

# `gaussian_per_basis=2` sets of `total_gaussians=20`: the shared ell_comps priors of
# the two bases are carried by the first Gaussian of each set.
TOTAL_GAUSSIANS = 20


@pytest.fixture(scope="module", autouse=True)
def push_config():
    """
    Point the library at this repository's ``config/``, which is where the mass and
    light profile prior defaults the model is built from come from.
    """
    from autolens import conf

    conf.instance.push(
        new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
    )


@pytest.fixture
def model(monkeypatch):
    """
    The ``vis_lp`` model, built at the shipped example tile's mask radius and centre.

    ``PYAUTO_SMALL_DATASETS`` is removed for the duration: it is the CI shortcut that
    collapses every MGE to two Gaussians in one basis, which would leave nothing to
    order and no second ell_comps pair to find.
    """
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    return initial_lens_model.vis_lp_model_from(
        mask_radius=MASK_RADIUS,
        dataset_centre=DATASET_CENTRE,
        redshift_lens=REDSHIFT_LENS,
        redshift_source=REDSHIFT_SOURCE,
    )


def _lens_ell_comps_1_priors(model):
    """
    The two lens bases' shared ``ell_comps_1`` priors, in basis order.

    Each basis shares one ``ell_comps_1`` prior across its 20 Gaussians, so the first
    Gaussian of each set carries it.
    """
    profile_list = model.galaxies.lens.bulge.profile_list

    return (
        profile_list[0].ell_comps.ell_comps_1,
        profile_list[TOTAL_GAUSSIANS].ell_comps.ell_comps_1,
    )


def _vector(model, overrides):
    """
    A physical parameter vector at every prior's median, with ``overrides`` (a list of
    ``(prior, value)`` pairs, matched by identity) substituted in.

    Ordered by prior id, which is the order ``assertions_satisfied_from_vector`` and
    ``instance_from_vector`` both assume. ``model.paths`` is not aligned with that
    order, which is why the priors are located by identity.
    """
    vector = []

    for _, prior in model.prior_tuples_ordered_by_id:
        value = next((v for p, v in overrides if p is prior), None)
        vector.append(prior.value_for(0.5) if value is None else value)

    return vector


def _arguments(model, vector):
    return {
        prior_tuple.prior: value
        for prior_tuple, value in zip(model.prior_tuples_ordered_by_id, vector)
    }


# ---------------------------------------------------------------------------
# Shape of the model
# ---------------------------------------------------------------------------


def test_prior_count_is_fifteen(model):
    """
    6 lens light (one shared centre, plus an independent ell_comps pair per basis),
    3 mass (ell_comps plus einstein_radius, the centre being fixed to the brightest
    pixel), 2 shear, 4 source light. The search's ``n_live=750`` is sized against this
    number, and the ordering assertion must not change it: an assertion removes prior
    volume, not parameters.
    """
    assert model.prior_count == 15


def test_exactly_one_assertion_is_attached(model):
    """
    ``order_bases=True`` with two bases attaches ``K - 1 = 1`` assertion, and it is
    attached to the ``Basis`` model rather than to this ``Collection``, so it is only
    visible through ``gathered_assertions``.
    """
    assert len(model.gathered_assertions()) == 1


# ---------------------------------------------------------------------------
# The two ell_comps boxes
# ---------------------------------------------------------------------------


def test_lens_ell_comps_priors_are_bounded_at_half(model):
    """
    Every one of the 40 lens Gaussians, in both bases. The box keeps the MGE out of the
    multi-blob regime that absorbs lensed-source flux into the lens light.
    """
    profile_list = model.galaxies.lens.bulge.profile_list

    assert len(profile_list) == 2 * TOTAL_GAUSSIANS

    for gaussian in profile_list:
        for prior in (gaussian.ell_comps.ell_comps_0, gaussian.ell_comps.ell_comps_1):
            assert prior.lower_limit == -LENS_ELL_COMPS_LIMIT
            assert prior.upper_limit == LENS_ELL_COMPS_LIMIT


def test_source_ell_comps_priors_are_bounded_at_zero_point_seven(model):
    """
    0.7^2 + 0.7^2 = 0.98 < 1: the largest axis-aligned box inside the unit disk, so
    ``|e| >= 1`` is unreachable by construction.
    """
    profile_list = model.galaxies.source.bulge.profile_list

    assert len(profile_list) == TOTAL_GAUSSIANS

    for gaussian in profile_list:
        for prior in (gaussian.ell_comps.ell_comps_0, gaussian.ell_comps.ell_comps_1):
            assert prior.lower_limit == -SOURCE_ELL_COMPS_LIMIT
            assert prior.upper_limit == SOURCE_ELL_COMPS_LIMIT


def test_each_lens_basis_has_its_own_ell_comps_prior(model):
    """
    The bases must stay independent: one prior per basis, shared across that basis's 20
    Gaussians, and not shared between the two. If they were shared there would be no
    degeneracy to order, and no ordering either.
    """
    ell_comps_1_a, ell_comps_1_b = _lens_ell_comps_1_priors(model)

    assert ell_comps_1_a is not ell_comps_1_b

    profile_list = model.galaxies.lens.bulge.profile_list

    for i in range(TOTAL_GAUSSIANS):
        assert profile_list[i].ell_comps.ell_comps_1 is ell_comps_1_a
        assert (
            profile_list[TOTAL_GAUSSIANS + i].ell_comps.ell_comps_1 is ell_comps_1_b
        )


# ---------------------------------------------------------------------------
# What the assertion accepts and rejects
# ---------------------------------------------------------------------------


def test_the_ordered_labelling_satisfies_the_assertion(model):
    """
    The first basis holding the larger ``ell_comps_1`` is the labelling kept.
    """
    ell_comps_1_a, ell_comps_1_b = _lens_ell_comps_1_priors(model)

    vector = _vector(model, [(ell_comps_1_a, 0.3), (ell_comps_1_b, -0.3)])

    assert bool(model.assertions_satisfied_from_vector(vector)) is True


def test_the_swapped_labelling_violates_the_assertion(model):
    """
    The same physical solution under the other labelling is the mode the assertion
    deletes. Nothing physical is lost: it is the same two ellipticities, exchanged.
    """
    ell_comps_1_a, ell_comps_1_b = _lens_ell_comps_1_priors(model)

    vector = _vector(model, [(ell_comps_1_a, -0.3), (ell_comps_1_b, 0.3)])

    assert bool(model.assertions_satisfied_from_vector(vector)) is False


def test_check_assertions_raises_on_the_swapped_labelling(model):
    """
    The exception form, which is what the NumPy path (``--use_cpu``) uses: Nautilus
    catches ``FitException`` and resamples.

    ``check_assertions`` is called on the ``Basis`` the assertion is attached to, not on
    the enclosing ``Collection``: it checks a model's *own* ``_assertions``, and on the
    NumPy path each model checks its own as its instance is built. (Only the traced path
    needs ``gathered_assertions``, which has no recursion to hook into.) The arguments
    are still built from the full model's prior ordering, since that is where the vector
    comes from.

    It is called directly rather than through ``instance_from_vector`` because that
    route is skipped entirely when a config sets ``general.test.exception_override``,
    which would let this test pass without the assertion doing anything.
    """
    import autofit as af

    ell_comps_1_a, ell_comps_1_b = _lens_ell_comps_1_priors(model)
    lens_bulge = model.galaxies.lens.bulge

    satisfying = _vector(model, [(ell_comps_1_a, 0.3), (ell_comps_1_b, -0.3)])
    violating = _vector(model, [(ell_comps_1_a, -0.3), (ell_comps_1_b, 0.3)])

    lens_bulge.check_assertions(_arguments(model, satisfying))

    with pytest.raises(af.exc.FitException):
        lens_bulge.check_assertions(_arguments(model, violating))
