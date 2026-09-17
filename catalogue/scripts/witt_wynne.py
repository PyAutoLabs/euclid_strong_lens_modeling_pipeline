"""
Euclid Catalogue: Witt-Wynne SIEP Projection
============================================

New here? Read ``start_here.py`` for the concepts (the model, the output layout,
what a fit writes) and ``catalogue/README.md`` for the run order.

Produces two per-lens products plus one master table:

- ``witt_wynne.in``   — the seven-line `isit4or2or1` input file, zero-centred,
  in gravlens conventions, ready to hand to the compiled ``SIEP_CLI.v1.0.cpp``.
- ``witt_wynne.csv``  — this lens's row of the master table.
- ``inspect/<sample>/witt_wynne.csv`` — the master table, one row per lens.

Each row is the lens's mass model projected onto a singular isothermal
elliptical **potential** (SIEP): an Einstein radius ``b``, a gravlens
ellipticity ``e``, a position angle East of North, the source's offset from the
lens centre, and — solved from those four numbers in closed form — the 4 / 2 / 1
verdict, the image positions, their signed magnifications and their time lags.

The point of the product is corroboration, not precision. When a transient
alert lands near a known Euclid lens, Schechter's LSST-broker code
(`isit4or2or1`) answers "could this be one of four images?" in microseconds
from exactly these numbers; publishing them per lens means a broker never has
to re-fit anything. ``docs/witt_wynne.md`` is the register for what the
projection does and does not preserve.

__What It Reads, And What It Cannot Know__

Everything comes out of one finished search per lens — by default the
``initial_lens_model/vis_pix`` stage, the same one ``lens_mass.csv`` and the
mass maps report, so every catalogue product describes the same tracer. From it:

- the **max-log-likelihood tracer** (``al.agg.TracerAgg``), whose tangential
  caustic is what the SIEP is matched to;
- the **dataset** (``al.agg.ImagingAgg``), which fixes the field of view the
  caustic is traced on;
- the **source position** from ``files/wcs.json`` (``agg.values("wcs")``).

No redshift is measured anywhere in this pipeline — every fitting script writes
the placeholders ``z_lens = 0.5``, ``z_source = 1.0`` because a single-plane
PyAutoLens model is dimensionless in them. ``--z_lens`` / ``--z_source``
therefore set the fiducial pair the distances and the lags are computed with,
and every row records ``redshift_source = placeholder`` so nothing downstream
mistakes them for measurements. The **verdict and the image positions do not
depend on them at all**; only ``d_ol``, ``d_ls`` and the lags do.

__The Source Position, And The Recomputed Fallback__

``files/wcs.json`` is the cheap route. ``util.wcs_dict_from`` writes
``source_centre_y_arcsec`` / ``source_centre_x_arcsec`` for a light-profile
source (``source_model == "light_profile"``) and a brightest-first
``source_clumps`` list for a pixelized one, whose first entry's peak is the
source position this producer uses. ``source_rule`` records which fired.

Fits written **before** those keys existed carry only the four WCS numbers, and
every DR1 tile fitted up to 2026-09-11 is in that state. Rather than drop them,
this producer rebuilds the maximum-likelihood ``FitImaging``
(``al.agg.FitImagingAgg``) and calls the *same* ``util`` helpers ``wcs.json``
itself uses — ``util.source_centre_from`` for a light-profile source,
``util.pixelized_source_clumps_from`` for a pixelized one — recording the rule
as ``recomputed_light_profile_centre`` / ``recomputed_brightest_clump_peak`` so
a row is never silently mixed with one read straight from the file. The
fallback costs roughly six seconds a lens; the ``wcs.json`` route is free.

A lens with neither (``source_model: "none"``, no clumps, nothing
reconstructed) is skipped with a message rather than written as a blank row:
without a source position there is nothing to solve.

__The Grid The Caustic Is Traced On__

The caustic-matched projection needs a grid to trace the tangential critical
curve on. The dataset's **masked** grid is the wrong one: ``LensCalc``'s
``evaluation_grid`` decorator rebuilds its own uniform grid from
``aa.Zoom2D(mask=grid.mask)``, and on the DR1 circular masks that route returns
a caustic whose semi-axes disagree with the converged answer by ~27 %
(measured on ``Tile102005065…``: ``e = 0.1267`` from the masked grid against
``e = 0.1009`` from every unmasked grid between 67x67 @ 0.1" and 400x400 @
0.01"). This producer therefore builds the cut-out's own **unmasked** uniform
grid — ``al.Grid2D.uniform(shape_native=dataset.data.shape_native,
pixel_scales=dataset.pixel_scales)`` — which is the regime the projection's
independent numerical review validated and in which ``b``, ``e`` and the
position angle are stable to ~1e-3 across every resolution tried.

``--caustic_pixel_scale`` (default 0.05") is the resolution the critical curve
is contoured at, passed straight to ``LensCalc``.

__Two Projections__

``--projection=caustic`` (the default) matches the SIEP astroid to the tracer's
own tangential caustic, so external shear and any secondary perturber are
folded in automatically. ``--projection=vector_sum`` is Schechter's literal
prescription — add the potential ellipticity and the shear as vectors in the
2-theta plane — which reads only the first admissible mass profile's
``ell_comps`` and so ignores perturbers. On a 136-case grid the review measured
the verdict agreeing with ``al.PointSolver`` 68/68 (caustic) against 66/68
(vector sum) inside the caustic, and 51/68 against 22/68 outside it. Ship the
caustic one; the other is there to reproduce the original paper's recipe.

__Sentinels, Not Exceptions__

``witt_wynne_util`` never raises on a model it cannot project. A tracer with no
Isothermal/PowerLaw-family mass profile, a sub-critical lens with no tangential
caustic, or a vector sum whose ellipticity and shear cancel returns a
**sentinel** ``WittWynne`` — ``valid=False``, NaN parameters and a ``reason``
string. The row is still written, with blank numeric cells, so the catalogue
records the lens and why it has no projection. The solver has its own sentinel:
a degenerate solve (a source on a potential axis, or exactly on a fold) gives
``n_images = -1`` and blank image cells on an otherwise valid row.

No ``.in`` file is written for a sentinel model — a file of ``nan`` fields
would be read by the C++ as data.

Stage 7 of ``scripts/build_inspection_bundle.sh``.

Usage
-----
    python catalogue/scripts/witt_wynne.py --sample=q1_walsmley

    python catalogue/scripts/witt_wynne.py \
        --sample=dr1_prelim_grade_ab \
        --projection=vector_sum \
        --inspect_dir=inspect/dr1_prelim_grade_ab
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import catalogue_util
import witt_wynne_util

# `util.py` lives at the project root and is imported inside the recompute
# fallback only, so `--help` never pays for the library imports it makes.
sys.path.insert(0, str(catalogue_util.PROJECT_ROOT))


MAXIMUM_IMAGES = 4

CSV_COLUMNS = [
    "lens_name",
    "projection",
    "valid",
    "reason",
    "b_arcsec",
    "e_gravlens",
    "pa_deg_E_of_N",
    "source_dx_arcsec",
    "source_dy_arcsec",
    "source_rule",
    "n_images",
    "image1_x",
    "image2_x",
    "image3_x",
    "image4_x",
    "image1_y",
    "image2_y",
    "image3_y",
    "image4_y",
    "mag1",
    "mag2",
    "mag3",
    "mag4",
    "lag1_days",
    "lag2_days",
    "lag3_days",
    "lag4_days",
    "z_lens",
    "z_source",
    "redshift_source",
    "d_ol_hinv_mpc",
    "d_ls_hinv_mpc",
    "h",
    "mass_profile",
    "search_name",
]


def source_centre_from_wcs(wcs_dict):
    """
    The source-plane ``(y, x)`` centre a fit's ``files/wcs.json`` records, and
    the rule that produced it.

    Two routes, in the order ``util.wcs_dict_from`` writes them:

    - ``source_model == "light_profile"`` — the source's light centre, written
      as ``source_centre_y_arcsec`` / ``source_centre_x_arcsec``
      (``light_profile_centre``);
    - otherwise the brightest clump of a pixelized reconstruction, whose
      ``source_clumps[0]`` carries ``peak_y_arcsec`` / ``peak_x_arcsec``
      (``brightest_clump_peak``).

    Returns ``(None, "none")`` when neither is present — an empty dict, a fit
    written before those keys existed, or ``source_model: "none"``.

    Every lookup is a ``.get``: PyAutoFit's JSON envelope **drops** ``None``
    values, so an unavailable number is an absent key rather than a null, and
    indexing would raise on exactly the lenses this has to survive.
    """
    wcs_dict = wcs_dict or {}

    if wcs_dict.get("source_model") == "light_profile":
        centre_y = wcs_dict.get("source_centre_y_arcsec")
        centre_x = wcs_dict.get("source_centre_x_arcsec")
        if centre_y is not None and centre_x is not None:
            return (float(centre_y), float(centre_x)), "light_profile_centre"

    clump_list = wcs_dict.get("source_clumps") or []

    if clump_list:
        peak_y = clump_list[0].get("peak_y_arcsec")
        peak_x = clump_list[0].get("peak_x_arcsec")
        if peak_y is not None and peak_x is not None:
            return (float(peak_y), float(peak_x)), "brightest_clump_peak"

    return None, "none"


def source_centre_recomputed_from(agg_query):
    """
    The source centre recomputed from the fit itself, for a result whose
    ``wcs.json`` predates the source keys.

    Rebuilds the maximum-log-likelihood ``FitImaging`` through
    ``al.agg.FitImagingAgg`` and calls the same two ``util`` helpers
    ``util.wcs_dict_from`` calls: ``source_centre_from`` (the source galaxy's
    light centre) and, when the source is pixelized and so has none,
    ``pixelized_source_clumps_from`` (the peak of the brightest clump of the
    reconstruction, found by thresholding against its 99th percentile with a
    brightest-mesh-pixel failsafe).

    Returns ``(centre, rule)`` with the rule prefixed ``recomputed_`` so a row
    built this way is never confused with one read straight out of the file,
    or ``(None, "none")`` when nothing was reconstructed.
    """
    import autolens as al
    import util

    fit_gen = al.agg.FitImagingAgg(aggregator=agg_query).max_log_likelihood_gen_from()
    fit = next(iter(fit_gen))[0]

    centre = util.source_centre_from(fit.tracer)

    if centre is not None:
        return centre, "recomputed_light_profile_centre"

    clump_list, _ = util.pixelized_source_clumps_from(fit)

    if clump_list:
        return (
            float(clump_list[0]["peak_y_arcsec"]),
            float(clump_list[0]["peak_x_arcsec"]),
        ), "recomputed_brightest_clump_peak"

    return None, "none"


def _cell(value, spec=".6f") -> str:
    """
    One CSV cell: the formatted number, or **blank** when it is missing or not
    finite.

    Blank rather than ``nan`` or a dropped column, so a sentinel row is the
    same width as every other and reads as missing in any CSV parser.
    """
    import math

    if value is None:
        return ""

    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""

    if not math.isfinite(value):
        return ""

    return format(value, spec)


def row_from(
    lens_name: str,
    model,
    images,
    projection: str,
    source_rule: str,
    search_name: str,
    mass_profile: str = "",
    redshift_source: str = "placeholder",
):
    """
    One ``witt_wynne.csv`` row, as a list of strings in ``CSV_COLUMNS`` order.

    Parameters
    ----------
    model
        The projected ``witt_wynne_util.WittWynne``. It is zero-centred here
        (``model.zeroed()``, which is idempotent), so no sky coordinate reaches
        the catalogue and every position is relative to the lens centre.
    images
        The ``(x, y, magnification, angle_deg)`` tuple of
        ``images_from_source`` **for the zero-centred model** — pass
        ``model.zeroed().images()``. Taken as an argument rather than
        recomputed so the row can be asserted against a hand-built solve.

    The four image slots are filled **in order of arrival**, i.e. by ascending
    time lag, so ``image1`` is always the leading image and ``lag1_days`` is
    always zero. Unused slots are blank, which is how a 2-image row differs
    from a 4-image one. A degenerate solve (``n_images == -1``) or a sentinel
    model leaves every image, magnification and lag cell blank while still
    recording the lens, the rule and the reason.
    """
    import numpy as np

    model = model.zeroed()

    x_image, y_image, magnification, _ = images
    n_images = witt_wynne_util.n_images_from(x_image, y_image)

    if n_images > 0:
        lags = np.asarray(model.lags(x_image, y_image), dtype=float)
        order = np.argsort(lags)
        x_image = np.asarray(x_image, dtype=float)[order]
        y_image = np.asarray(y_image, dtype=float)[order]
        magnification = np.asarray(magnification, dtype=float)[order]
        lags = lags[order]
    else:
        x_image = y_image = magnification = lags = np.asarray([], dtype=float)

    def slot(values, index, spec=".6f"):
        return _cell(values[index], spec) if index < values.size else ""

    row = [
        lens_name,
        projection,
        str(bool(model.valid)),
        model.reason,
        _cell(model.b),
        _cell(model.e),
        _cell(model.pa_deg, ".4f"),
        _cell(model.source[0]),
        _cell(model.source[1]),
        source_rule,
        str(int(n_images)),
    ]
    row += [slot(x_image, i) for i in range(MAXIMUM_IMAGES)]
    row += [slot(y_image, i) for i in range(MAXIMUM_IMAGES)]
    row += [slot(magnification, i) for i in range(MAXIMUM_IMAGES)]
    row += [slot(lags, i, ".4f") for i in range(MAXIMUM_IMAGES)]
    row += [
        _cell(model.z_lens, ".4f"),
        _cell(model.z_source, ".4f"),
        redshift_source,
        _cell(model.d_ol, ".4f"),
        _cell(model.d_ls, ".4f"),
        _cell(model.h, ".6g"),
        mass_profile,
        search_name,
    ]

    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project each lens's mass model onto Witt-Wynne SIEP space and "
            "write the isit4or2or1 input file and catalogue row for it."
        )
    )
    catalogue_util.add_common_arguments(parser)
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="initial_lens_model",
        help="Pipeline stage the results were written under.",
    )
    parser.add_argument(
        "--search_name",
        metavar="name",
        default="vis_pix",
        help=(
            "Keep only fits whose search.name matches. Default: 'vis_pix', the "
            "stage whose mass model lens_mass.csv reports."
        ),
    )
    parser.add_argument(
        "--z_lens",
        metavar="z",
        type=float,
        default=0.5,
        help=(
            "Fiducial lens redshift used for the distances and time lags. "
            "Default: 0.5, the placeholder every fitting script uses. Recorded "
            "with redshift_source=placeholder; the verdict and the image "
            "positions do not depend on it."
        ),
    )
    parser.add_argument(
        "--z_source",
        metavar="z",
        type=float,
        default=1.0,
        help=(
            "Fiducial source redshift. Default: 1.0, the placeholder every "
            "fitting script uses."
        ),
    )
    parser.add_argument(
        "--projection",
        metavar="name",
        default="caustic",
        choices=["caustic", "vector_sum"],
        help=(
            "How the tracer is projected onto the SIEP. 'caustic' (default) "
            "matches the SIEP astroid to the tracer's tangential caustic and "
            "folds in shear and perturbers; 'vector_sum' is Schechter's "
            "literal ellipticity-plus-shear prescription."
        ),
    )
    parser.add_argument(
        "--caustic_pixel_scale",
        metavar="arcsec",
        type=float,
        default=0.05,
        help=(
            "Resolution the tangential critical curve is contoured at. "
            "Default: 0.05."
        ),
    )
    return parser.parse_args()


def _distances_from(cosmology, z_lens: float, z_source: float):
    """
    ``(d_ol, d_ls, h)`` in ``h^-1 Mpc`` for a redshift pair, on the tracer's own
    cosmology.

    The same conversion ``witt_wynne_util._distances_from`` makes, applied to
    the redshifts the *command line* asked for rather than the ones the tracer
    carries. The tracer's are the pipeline's dimensionless placeholders, so this
    is what makes ``--z_lens`` / ``--z_source`` mean anything.
    """
    h = float(cosmology.H0) / 100.0
    d_ol = cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=z_lens)
    d_ls = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=z_lens, redshift_1=z_source
    )
    return float(d_ol) / 1e3 * h, float(d_ls) / 1e3 * h, h


def _with_redshifts(model, cosmology, z_lens: float, z_source: float):
    """
    The same model with the command line's redshifts and the distances that
    follow from them.

    A sentinel model keeps its NaN parameters; only the four redshift/distance
    fields and ``h`` are replaced, so a row that could not be projected still
    records the fiducial cosmology it would have used.
    """
    from dataclasses import replace

    d_ol, d_ls, h = _distances_from(
        cosmology=cosmology, z_lens=z_lens, z_source=z_source
    )

    return replace(model, d_ol=d_ol, d_ls=d_ls, z_lens=z_lens, z_source=z_source, h=h)


def main():
    """
    __One Aggregator Per Lens__

    Like ``lens_mass_maps.py`` and ``deblending.py``, this producer loops over
    lens directories and builds an aggregator scoped to each: the ``.in`` file
    is per lens, and the source position has to be paired with the tracer it
    was measured from, which a whole-sample aggregator cannot guarantee.

    Failures are per lens too. A lens with no completed search under the tag
    and search name, one whose ``wcs.json`` records no source, one whose
    reconstruction is empty: all are printed and skipped, because
    ``scripts/build_inspection_bundle.sh`` runs under ``set -e`` and a raised
    exception here would abort every later stage over a single unusable fit.
    """
    args = parse_args()
    output_path, inspect_path = catalogue_util.resolve_paths(args)

    from autofit.aggregator.aggregator import Aggregator
    import autolens as al

    sample_root = catalogue_util.sample_root_from(output_path, args.sample)
    dataset_name_list = catalogue_util.dataset_names_from(sample_root)

    project = (
        witt_wynne_util.witt_wynne_from_tracer
        if args.projection == "caustic"
        else witt_wynne_util.witt_wynne_vector_sum
    )

    rows = []

    for dataset_name in dataset_name_list:

        print(dataset_name)

        try:
            agg = Aggregator.from_directory(
                directory=sample_root / dataset_name,
                completed_only=True,
                unzip_temporary=True,
            )

            agg_query = agg.query(agg.unique_tag == args.unique_tag)
            if args.search_name is not None:
                agg_query = agg_query.query(agg_query.search.name == args.search_name)

            tracer = next(
                iter(
                    al.agg.TracerAgg(aggregator=agg_query).max_log_likelihood_gen_from()
                )
            )[0]

            dataset = next(
                iter(al.agg.ImagingAgg(aggregator=agg_query).dataset_gen_from())
            )[0]

            """
            __The Source Position__

            The `wcs.json` route first, then the recompute. Both are per lens,
            and the rule that fired is a column, so a catalogue built over a
            tree of mixed vintages says per row where its source came from.
            """
            wcs_list = list(agg_query.values("wcs"))
            source_centre, source_rule = source_centre_from_wcs(
                wcs_list[0] if wcs_list else {}
            )

            if source_centre is None:
                source_centre, source_rule = source_centre_recomputed_from(agg_query)

            if source_centre is None:
                print(
                    f"skipping {dataset_name}: no source position in wcs.json and "
                    "none could be recomputed from the fit"
                )
                continue

            """
            __The Projection__

            The caustic is traced on the cut-out's own unmasked uniform grid —
            see "__The Grid The Caustic Is Traced On__" in the module
            docstring for why it is not `dataset.grid`.
            """
            grid = al.Grid2D.uniform(
                shape_native=dataset.data.shape_native,
                pixel_scales=dataset.pixel_scales,
            )

            model = project(
                tracer=tracer,
                grid=grid,
                source_centre=source_centre,
                caustic_pixel_scale=args.caustic_pixel_scale,
            )

            model = _with_redshifts(
                model=model,
                cosmology=tracer.cosmology,
                z_lens=args.z_lens,
                z_source=args.z_source,
            )

            mass, _ = witt_wynne_util._mass_and_shear_from(tracer)
            mass_profile = "" if mass is None else type(mass).__name__

            zeroed = model.zeroed()
            images = zeroed.images()

            if model.valid:
                witt_wynne_util.write_isit_input(
                    path=inspect_path / dataset_name / "witt_wynne.in",
                    model=model,
                    zero_centre=True,
                )
            else:
                print(f"{dataset_name}: no .in written — {model.reason}")

            rows.append(
                row_from(
                    lens_name=dataset_name,
                    model=zeroed,
                    images=images,
                    projection=args.projection,
                    source_rule=source_rule,
                    search_name=args.search_name,
                    mass_profile=mass_profile,
                )
            )

        except (ValueError, FileNotFoundError, KeyError, IndexError) as e:
            print(f"skipping {dataset_name}: {e}")
            continue

    if not rows:
        print("no lenses projected; no witt_wynne.csv written")
        return

    """
    __The Master CSV And Its Per-Lens Split__

    Written with `csv.writer` rather than `af.AggregateCSV`: none of these
    columns is a model parameter with a posterior, so there is no aggregator
    column type for them. `lens_name` is the first column because
    `catalogue_util.write_per_tile_csv` groups the master on it to drop each
    lens's own row into its own folder.
    """
    master_csv = inspect_path / "witt_wynne.csv"

    with open(master_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)

    written = catalogue_util.write_per_tile_csv(
        master_csv=master_csv,
        inspect_path=inspect_path,
        filename="witt_wynne.csv",
    )

    print(f"{len(rows)} rows -> {master_csv} ({written} per-lens copies)")


if __name__ == "__main__":
    main()
