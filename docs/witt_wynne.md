# The Witt–Wynne SIEP projection (`witt_wynne.csv`, `witt_wynne.in`)

Register for stage 7 of the inspection bundle. What it publishes, in which
conventions, how accurate it is, and how to run it. It is not a tutorial — the
maths lives in `catalogue/scripts/witt_wynne_util.py`'s docstrings and the
papers it cites.

---

## What the projection is, and why it exists

The singular isothermal elliptical **potential** (SIEP) is the one strong-lens
model whose lens equation collapses to a quartic with a closed-form solution.
Four numbers — an Einstein radius `b`, an ellipticity `e`, a position angle, and
the source's offset from the lens centre — give you, without iteration, the
number of images (4, 2 or 1), where they are, their signed magnifications and
their relative time lags. A solve costs about 60 µs.

That speed is the whole point. Paul Schechter's `isit4or2or1` exists so an LSST
alert broker can answer "is this transient a fourth image of a known quad, or a
foreground star?" before the object fades. The corroboration it is designed for
is the one Schechter, Lu & Hernández performed on **SN 2025wny**. Publishing the
projection per lens means a broker holding a Euclid lens catalogue never has to
re-fit anything: it reads four numbers and solves.

This pipeline fits SIE + external shear mass models with MGE lens light and
either a light-profile or a pixelized source. Stage 7 takes the
maximum-log-likelihood tracer of the `initial_lens_model/vis_pix` search — the
same one `lens_mass.csv` and the mass maps describe — and projects it onto the
SIEP.

Two projections are available:

- **`--projection=caustic`** (the default) fits the SIEP astroid to the tracer's
  own tangential caustic. Because a caustic is a property of the whole tracer,
  external shear and any secondary perturber are folded in automatically.
- **`--projection=vector_sum`** is Schechter's literal prescription: add the
  potential ellipticity and the shear as vectors in the 2θ plane and discard
  the perpendicular components. It reads only the first admissible mass
  profile's `ell_comps`, so it ignores perturbers.

Ship the caustic one. The vector sum is kept to reproduce the original recipe
and to make the difference between them measurable.

---

## Conventions

`isit4or2or1` is a *gravlens*-convention code. Three of its conventions differ
from PyAutoLens's and are the usual source of a silently wrong answer.

| Quantity | `isit4or2or1` / `witt_wynne.csv` | PyAutoLens |
|---|---|---|
| Ellipticity | `e = 1 − q_ψ`, of the **potential** | `ell_comps`, magnitude `(1 − q)/(1 + q)`, of the **density** |
| Position angle | `PA = (θ_ccw + 90) mod 180`, degrees **East of North** | `angle`, degrees counter-clockwise from the positive x-axis |
| Coordinate order | `(x, y)` | `(y, x)` |
| Distances | angular diameter, **h⁻¹ Mpc** | kpc, on the tracer's cosmology |
| Origin | **zero-centred**: the lens is at `(0, 0)` | sky / cut-out frame |

The density→potential ellipticity map used by the vector sum is the documented
first-order one, `e ≈ (1 − q)/3`; the caustic-matched projection needs no such
map because it fits `e` to the caustic directly.

**Time lags** are in days and referred to the leading image, so `lag1_days` is
always exactly zero. The time constant keeps the C++'s own rounded literals
(`D_H = 3000`, `9.78e9/h` yr, `365.0` d/yr), which together run **0.12 % low**
against the exact `(1 + z_l) D_l D_s / (c D_ls)`. That is deliberate: it makes
these lags directly comparable with `isit4or2or1`'s. The C++ additionally fixes
`h = 0.7` in its time constant regardless of the distances it is given, so lags
it computes from a `witt_wynne.in` written on a Planck15 cosmology come out a
factor `h/0.7` smaller — scale by `0.7/h` (the `h` column) to compare.

**No sky coordinates are published.** `write_isit_input(..., zero_centre=True)`
translates the lens to the origin and the source with it. The solver is exactly
translation invariant (positions, magnifications and lags all change by
0.000e+00), so nothing is lost, and the file can be shared without divulging
where a Euclid source sits. The CSV's `source_dx_arcsec` / `source_dy_arcsec`
and every `image*_x` / `image*_y` are in that same zero-centred frame.

---

## How accurate it is

From the independent numerical review of 2026-09-17 (posted in full on
[issue #84](https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/issues/84)),
which used `al.PointSolver` and `tracer.time_delays_from` as oracles over a
136-case grid — axis ratio `q ∈ {0.5, 0.7, 0.85, 0.95}` × shear
`γ ∈ {0, 0.03, 0.06, 0.1, 0.15}` × misalignment `{0, 30, 60, 90}°`, with sources
at 0.5× and 1.02× the caustic:

| | verdict, source **inside** the caustic | verdict, source **outside** | median / max Δposition | max Δlag |
|---|---|---|---|---|
| caustic-matched | **68/68** | **51/68** | 0.136" / 0.398" | 32.6 d |
| vector sum | 66/68 | 22/68 | 0.148" / 0.409" | 47.8 d |

Read that as: **inside the caustic the verdict is reliable and the positions are
good to 0.07–0.4"; outside it the verdict is not.** Every outside-caustic
disagreement is in one direction — a true 2 reported as a 4 — because the
projected astroid is slightly too large. Time lags are 5–13 % of their own span.

The solver itself is exact where it is well conditioned: away from the potential
axes it reproduces an independent brute-force Newton solve to **4.5e-13 arcsec**
in position, 5.4e-13 relative in magnification and 2.3e-13 days in lag.

The astroid fit degrades with flattening: its residual on the two caustic
semi-axes is −10 % / +11 % at `q = 0.5` and −18 % / +16 % at `q ≤ 0.35`.

### Measured on ten real DR1 models, 2026-09-17

The review's grid was synthetic and used `b ≈ 1.19"`. Running this producer over
the ten `dr1_prelim_grade_ab` tiles (`b` from 0.49" to 3.03", `e` from 0.043 to
0.32, source = the brightest clump of each pixelized reconstruction) and
cross-checking every verdict against `al.PointSolver` on the same tracer and the
same source position:

- **verdict 9/10.** The one disagreement is `Tile102007903…` (`b = 1.93"`,
  `e = 0.302`, source outside the astroid): the SIEP says 2, `PointSolver`
  says 4. Note that is the *opposite* direction from the review's
  outside-caustic bias, and the vector-sum projection got it right on that tile.
- **positions: median 0.10"–1.70" per tile, worst 2.70"** — but nine of the ten
  sources fall **outside** the astroid, and the error scales with the Einstein
  radius. As a fraction of `b` it is 10–40 %, which is the same ratio the review
  measured (0.136" median on `b ≈ 1.19"`).
- A controlled sweep at 0.3, 0.5, 0.9, 1.5 and 3.0× the astroid's short
  semi-axis on two of those tiles gave median 0.17"–0.30" on the `b = 0.55"`
  lens and 0.71"–1.22" on the `b = 3.03"` one, and on the `b = 0.55"` lens the
  SIEP said **4** inside the astroid where `PointSolver` said **2**.

Read that as: **treat the numbers as fractions of the Einstein radius, not as
arcseconds, and treat the verdict as indicative on a real shear-bearing Euclid
model rather than as the 68/68 the synthetic grid suggests.** They are what they
are meant to be — a fast corroboration to be checked against a real solve, not a
substitute for one.

---

## Sentinels: `valid`, `reason`, `n_images = -1`

Nothing in this stage raises. Two independent failure channels are recorded
instead, and a lens is never dropped from the catalogue because of either.

- **`valid = False`** means the *projection* could not be made. `reason` says
  why: no Isothermal/PowerLaw-family mass profile in the tracer, no tangential
  caustic (a sub-critical lens on this grid), or — vector sum only — an
  ellipticity and shear that cancelled (`e < 1e-3`, which happens exactly when
  `e_potential = γ` and the two are aligned). `b`, `e`, `pa_deg_E_of_N` and every
  derived cell are blank. **No `witt_wynne.in` is written**, because a file of
  `nan` fields would be read by the C++ as data.
- **`n_images = -1`** means the *solve* was degenerate on an otherwise valid
  model: the source lies on a potential axis or at the lens centre
  (`min(|p|, |q|) < 1e-6`), `e ∉ (0, 1)`, or a position, magnification or angle
  came back non-finite (a source exactly on a fold or cusp). The image,
  magnification and lag cells are blank. It can never be confused with a genuine
  1-image verdict, whose position is finite.

A `reason` on a `valid = True` row records a choice the projection had to make —
at present only "there were two admissible mass profiles and the first was
used".

Missing numbers are written as **blank cells**, never as `nan` and never by
dropping a column, so every row is the same width.

---

## Caveats

**The redshifts are placeholders.** No redshift is measured anywhere in this
pipeline. Every fitting script writes `z_lens = 0.5`, `z_source = 1.0` because a
single-plane PyAutoLens model is dimensionless in them, and this producer's
`--z_lens` / `--z_source` default to the same pair. Every row therefore carries
`redshift_source = placeholder`. **The verdict and the image positions do not
depend on them at all**; `d_ol_hinv_mpc`, `d_ls_hinv_mpc` and every `lag*_days`
do, and are fiducial numbers rather than measurements. A photometric redshift
is a downstream product of the SED chain; when one exists, pass it and change
the column.

**The SIEP has no shear.** Schechter's Zenodo v1.0 models a lens with an
elliptical potential and nothing else. The pipeline's mass model is SIE +
external shear. The caustic-matched projection folds the shear into `e` and the
position angle through the caustic it fits — which is why it is the default —
but the result is still a *single* elliptical potential, and any information in
the shear that an astroid cannot carry is gone. The vector sum is worse in this
respect: it adds the two as vectors and keeps only the sum.

**Outside the caustic the verdict is not reliable** — see the table above. A
transient whose projected source position falls outside the astroid should be
treated as "2 or fewer", not as the number in the column.

**The source position is a reference value, not the transient's.** It is the
lens's own source: the light-profile centre for a `vis_lp`-style fit, or the
peak of the brightest clump of a pixelized reconstruction, recorded in
`source_rule`. Schechter's protocol fits the *transient's* position, so a broker
should substitute its own and re-solve. `witt_wynne_util.source_from_image`
inverts one detected image back to a source for exactly that.

**`source_rule` may say `recomputed_*`.** Fits written before 2026-09-12 carry a
`files/wcs.json` with only the four WCS numbers. For those, the producer rebuilds
the maximum-likelihood `FitImaging` and calls the same `util` helpers `wcs.json`
itself uses, recording `recomputed_light_profile_centre` or
`recomputed_brightest_clump_peak`. The value is the same quantity; the rule name
says it came from the fallback, which costs a few seconds a lens.

**The caustic is traced on the cut-out's unmasked grid.** `LensCalc` rebuilds its
own evaluation grid from `aa.Zoom2D(mask=grid.mask)`, and on the DR1 circular
masks that route returns caustic semi-axes ~27 % different from the converged
answer (measured on `Tile102005065…`: `e = 0.1267` masked against `e = 0.1009`
from every unmasked grid between 67×67 @ 0.1" and 400×400 @ 0.01"). The producer
therefore builds `al.Grid2D.uniform` from the dataset's shape and pixel scale,
which is the regime the review validated.

---

## The `.in` file

Seven whitespace-separated lines, read by the C++ in this order:

```
x_lens   y_lens          # always 0 0 — zero-centred
x_source y_source        # arcsec, relative to the lens
ellipticity              # e = 1 - q_psi, of the potential
einstein_radius          # b, arcsec
position_angle           # degrees East of North, 0 <= PA < 180
D_ol D_ls                # angular diameter distances, h^-1 Mpc
z_lens z_source          # the fiducial placeholders
```

The CSV carries the same numbers to written precision (`%.6f` costs 2.7e-7),
plus the solved verdict, positions, magnifications and lags.

> **Not verified against the compiled C++.** `SIEP_CLI.v1.0.cpp` was not
> available when the review ran, so the field order is right per the documented
> spec but the round-trip has never been executed. Anyone with the Zenodo
> archive to hand should do it and report back on issue #84.

---

## How a colleague runs it

```bash
# 1. Install PyAutoLens and clone the pipeline.
pip install --upgrade pip
pip install autolens[coolest]
git clone https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline
cd euclid_strong_lens_modeling_pipeline

# 2. Fit the one Q1 lens this repository ships.
#    ~10 min on a GPU, ~20 min on 8 CPU cores.
python scripts/initial_lens_model.py \
    --sample=q1_walsmley \
    --dataset=102018665_NEG570040238507752998

# 3. Project it. Seconds per lens.
python catalogue/scripts/witt_wynne.py --sample=q1_walsmley
# -> inspect/q1_walsmley/witt_wynne.csv
# -> inspect/q1_walsmley/102018665_NEG570040238507752998/witt_wynne.in

# 3b. Or build the whole 9-stage inspection bundle, of which this is stage 7.
SKIP_SED=1 CREATE_ARCHIVE=0 bash scripts/build_inspection_bundle.sh q1_walsmley
```

Then hand the `.in` to `isit4or2or1`:

```bash
# Download SIEP_CLI.v1.0.cpp from Zenodo, DOI 10.5281/zenodo.20086659
g++ -std=c++17 -O2 -o siep_cli SIEP_CLI.v1.0.cpp
./siep_cli < inspect/q1_walsmley/102018665_NEG570040238507752998/witt_wynne.in
```

Its positions and magnifications should reproduce the CSV's exactly; its lags
come out a factor `h/0.7` smaller (see **Conventions**).

Useful flags: `--projection=vector_sum` for Schechter's literal recipe,
`--search_name=vis_lp` to project the light-profile-source stage instead,
`--z_lens` / `--z_source` for a different fiducial pair, `--caustic_pixel_scale`
for the critical-curve resolution. `--help` documents them all and costs no
library import.

---

## Attribution

The solver is a derivative work of **`isit4or2or1` v1.0** by Paul L. Schechter,
Lu & Hernández, archived at Zenodo under **DOI
[10.5281/zenodo.20086659](https://doi.org/10.5281/zenodo.20086659)** and released
under **CC-BY-4.0**. The same CC-BY-4.0 attribution applies to this port. Please
cite the Zenodo record, and:

- Schechter, Lu & Hernández 2026 — `isit4or2or1` v1.0, SN 2025wny and the LSST
  alert protocol.
- Witt 1996, ApJ 472, L1 — the hyperbola the image positions lie on.
- Wynne & Schechter 2018, arXiv:1808.06151 — the ellipse that intersects it.
- Schechter & Wynne 2019, arXiv:1901.08517 — the resulting quartic.
- Falor & Schechter 2022, arXiv:2205.06269 — the asymptotically circular lens
  equation and the 4/2/1 root count.
