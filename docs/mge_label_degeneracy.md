# The Two-Fold Label Degeneracy in the Euclid Lens-Light MGE

- **Date**: 2026-09-08
- **Issue**: [#54](https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/issues/54)
- **Status**: implemented; see section 7
- **Scope**: research note plus its implementation record. Sections 1 to 6 record the mechanism,
  the measurements that establish it, and the option matrix as they stood when nothing had been
  built; section 7 records what was built and where it departed from that matrix.

## 1. Symptom and evidence

Comparing the May run of the DR1 tiles (the `euclid` project, RAL job 306843) against the
September run (`euclid_dr1_prelim`, RAL job 342301) on byte-identical data, 6 of the 8
well-determined tiles show the two lens-light MGE sets exchanged. Tile 102005065 is the clearest
case: set A's position angle moves from 0 to 90 degrees and set B's from 90 to 0. In every tile
the *unordered pair* of ellipticities is conserved, and the mass ell_comps, the centres, the
Einstein radii and the log evidence are unchanged. Read one set at a time, this looks like a sign
flip or a genuine change in the inferred lens light. It is neither.

A session scratch script (Appendix A) reproduced the underlying symmetry directly, on the real
shipped example tile `q1_walsmley/102018665_NEG570040238507752998`. Building the exact model
`scripts/initial_lens_model.py:162-236` composes, then evaluating the analysis at random points in
the prior box and again with the two sets' ell_comps swapped, the log likelihood is unchanged:
differences of 0.0, 1.8e-12 and 0.0 on the NumPy backend and exactly 0.0 on all three JAX
evaluations. The 1.8e-12 is floating-point summation order, not a real difference.

## 2. Mechanism

The lens light is `al.model_util.mge_model_from(..., total_gaussians=20, gaussian_per_basis=2)`
(`scripts/initial_lens_model.py:162-168`), giving a Basis of 40 linear Gaussians as two sets of
20. Three properties make the two sets exchangeable:

1. **The sigma ladders are identical, value for value.** `mge_model_from` computes one
   `log10_sigma_list` before the basis loop
   (`PyAutoGalaxy/autogalaxy/analysis/model_util.py:119`) and reuses it for every basis, so both
   sets run from 1e-4 to the mask radius over the same 20 values. The demo confirms
   `set A sigma ladder == set B sigma ladder: True`, with a first sigma of 1e-4 and a last of 3.5
   in both. These are `af.Constant`, not priors, so they contribute no non-linear parameters and
   cannot distinguish the sets.
2. **The ell_comps priors are identically distributed and independent.**
   `_make_ell_comps_priors()` (`model_util.py:146-172`) is called once per basis, so each set gets
   its own prior pair drawn from the same distribution. The pipeline then replaces both pairs with
   its own tighter `TruncatedGaussianPrior(mean=0.0, sigma=0.3, lower_limit=-0.5, upper_limit=0.5)`
   in the loop at `scripts/initial_lens_model.py:179-189`, preserving that structure exactly. The
   demo shows four distinct prior ids, two per set, each shared across 20 Gaussians.
3. **The centre is shared and the intensities are linear.** Both sets sit at the same centre, and
   the 40 intensities are solved jointly at every likelihood evaluation rather than sampled.

Swapping the two sets' ell_comps therefore permutes columns of the linear design matrix. The
column set is unchanged, so the solved intensities are the same values in permuted order and the
model image is bit-identical. This is an exact symmetry of the likelihood, not an approximate one,
and the prior is symmetric under the same swap. The posterior consequently has two exactly equal
modes. `af.Nautilus` is constructed without a `seed` (`scripts/initial_lens_model.py:266-272`;
`seed` defaults to `None` at `PyAutoFit/autofit/non_linear/search/nest/nautilus/search.py:103`),
so which set lands in which mode is a coin flip per run.

## 3. What it does and does not affect

**Affected.** The per-set marginal posteriors are two-mode mixtures. A single set's position angle
or ellipticity from an unordered run is therefore not a citable quantity; only the unordered pair
is. Unseeded reruns are not reproducible at the parameter level, even with identical data and
identical library versions.

**Not affected.** The fit quality is identical by construction. `vis_pix` freezes the `vis_lp`
maximum-likelihood lens light as an instance (`scripts/initial_lens_model.py:603-604`, explained
at `:559-566`), so the downstream fit sees the same 40 Gaussians with the same intensities
whichever labelling won; only the frozen numbers, and any identifier that hashes them, differ.
Mass ell_comps, centres, Einstein radii and log evidence are unaffected, which is exactly what the
May/September comparison found.

**The log evidence is a weak diagnostic here.** Imposing an ordering halves the prior volume but
renormalises over that half, so an ordered model has the same log Z as an unordered one whose
sampler visited both modes. A run that visited only one mode is low by ln 2 = 0.69. The prompt's
witness accepts log Z within 1.0 of the 9492.7 baseline, so it cannot distinguish "both modes
found" from "one mode found". The witness should therefore also compare the **per-set marginal
widths**: a bimodal marginal is visibly wide, and it collapsing is the sharper evidence that the
symmetry is gone.

## 4. Options

| | exact? | fit quality | run time | log Z | vis_pix propagation | identifier | reproducibility | belongs |
|---|---|---|---|---|---|---|---|---|
| (a) ordering assertion | yes | unchanged | ~unchanged | unchanged | frozen values now canonical | changes | full | pipeline after the PyAutoFit fix, library later |
| (b) different sigma ladders | no (offset) / model change (split) | changed | changed | changed | changed | changes | partial | rejected |
| (c) explicit Nautilus seed | no | unchanged | unchanged | unchanged | unchanged | changes | run-to-run only | pipeline, with (a) |
| (d) post-hoc relabelling | yes, at read time | unchanged | none | unchanged | unchanged | none | reporting only | results layer |

The option matrix below is the state of the question as of the note's date; section 7
records which options were implemented and how the ordering key changed on the way.

**(a) Ordering assertion.** After the `af.Collection` is composed, attach
`model.add_assertion((eA0**2 + eA1**2) > (eB0**2 + eB1**2))`, following the precedent at
`autolens_workspace/scripts/guides/modeling/cookbook.py:311-317`. This removes the symmetry
exactly, leaves the likelihood untouched, and should cost little run time: half the prior volume
is rejected and the rejected region is a smooth half-space that Nautilus's bounds learn quickly.
The demo confirms it attaches cleanly, with `prior_count` still 15 and one assertion on the model.

**It cannot run on this pipeline's JAX path today.** `Fitness.call`
(`PyAutoFit/autofit/non_linear/fitness.py:380-394`) catches `exc.FitException` and returns the
resample sentinel only on the NumPy branch; the JAX branch has no such `except`. The demo shows
all three behaviours: NumPy returns -1e+99 for a violating vector; the non-vmapped JAX `Fitness`
lets `FitException` escape from `check_assertions`
(`PyAutoFit/autofit/mapper/prior_model/abstract.py:193-226`, reached from
`instance_for_arguments` at `:1600-1628`), which would kill the run at the first violating sample;
and the vmapped `Fitness` that Nautilus actually builds (`use_jax_vmap=True` by default,
`nautilus/search.py:116, 242-256`) raises `jax.errors.TracerBoolConversionError` at trace time for
satisfying and violating vectors alike, so the assertion cannot even compile. Option (a) is
therefore gated on a PyAutoFit change: evaluate `_assertions` with the array module inside the
traced path and fold the result into the log likelihood with `xp.where` onto the resample
sentinel, rather than raising. That is precisely the penalty term the traced sibling
`__model_constraint__` anticipates in its module docstring
(`PyAutoFit/autofit/mapper/prior_model/constraint.py`), whose non-negative violation measure is
consumed today only by diagnostic counters. Until then (a) works only under `--use_cpu`.

A sibling variant, (a') reparametrising instead of rejecting (for example `eB1 = eA1 - delta` with
`delta` positive), is rejected: it distorts set B's prior, can push B outside the +/-0.5 box the
pipeline deliberately imposes, and changes the evidence.

**(b) Different sigma ladders.** Offset or interleaved ladders over the same radial range leave
the two sets near-exchangeable: the degeneracy becomes approximate rather than exact, which is
worse than either extreme, because two near-equal modes still split the posterior but no longer
cancel cleanly. Radially split ladders (one set inner, one outer) do break it, but they change the
physics, allowing an isophote twist with radius that the current model forbids, and they change
the evidence. That is a model change, not a degeneracy fix. Rejected.

**(c) Explicit seed.** Pass `seed=<int>` to `af.Nautilus` at
`scripts/initial_lens_model.py:266-272`. Note that the prompt's suggested location,
`config/non_linear/nest.yaml`, does not exist: the pipeline's `config/non_linear/` holds only
`GridSearch.yaml`, and there is no `nest.yaml` anywhere in PyAutoFit. `seed` is a search
identifier field (`nautilus/search.py:84`), so setting it changes the run identifier. It buys
reproducibility only; the posterior stays bimodal and any library or data change can still flip
the labelling. Cheap, and worth applying in the same edit as (a).

**(d) Post-hoc canonical relabelling.** Order the two sets by ellipticity magnitude wherever
results are read (results layer, aggregator, CSV export). This fixes the reading of runs already
on disk, changes nothing about the sampler, the evidence or the identifier, and requires no rerun.
It does not remove the bimodality, so per-set marginal *widths* stay inflated even though the
reported point estimates become stable.

**Placement.** The symmetry is created by `mge_model_from` for every user who passes
`gaussian_per_basis > 1` (`PyAutoGalaxy/autogalaxy/analysis/model_util.py:7`), not by anything
specific to this pipeline. The ordering therefore belongs upstream as an option, once PyAutoFit
can enforce it under JAX. The pipeline should apply it locally first.

## 5. Recommendation and rollout

1. **PyAutoFit.** Make assertions traceable: evaluate them with the array module inside
   `Fitness.call`'s JAX branch and fold the violation into the log likelihood via `xp.where` onto
   the resample sentinel, with a test that pins the vmapped path. This is the blocking step.
2. **Pipeline.** Add the ordering assertion and an explicit `seed` to the `vis_lp` search in
   `scripts/initial_lens_model.py`. Both change the run identifier and so force fresh runs; land
   them **between phases only**. `euclid_dr1_prelim` phase 4 is live on RAL as of this note.
3. **Witness.** Two independent unseeded `vis_lp` runs on tile 102005065 must give set-A and set-B
   ell_comps agreeing run-to-run to 0.05 per component, with log Z within 1.0 of the 9492.7
   baseline, **plus** the per-set marginal-width check from section 3, which is the part that
   actually separates "ordered" from "one mode found".
4. **PyAutoGalaxy.** Add an ordering option to `mge_model_from` for `gaussian_per_basis > 1`, so
   every user gets the fix rather than this pipeline alone.
5. **Meanwhile.** Nothing. Decision 2026-09-08: the tiles will be rerun from scratch once (1)
   and (2) land, which also gives the speed comparison, so (d) is not implemented.

Steps 1, 2 and 4 have since landed; section 7 records what was built, and why the ordering key
is not the magnitude this section assumed.

## 6. Interim guidance for reading current results

Compare the **unordered pair** of ellipticities between runs; it is conserved. Never cite one
set's position angle or ellipticity from an unordered run, and do not report a change in one set
between two runs as a physical result without first checking whether the pair as a whole moved.
The mass model, the source, the centres, the Einstein radii and the log evidence are unaffected
and can be read as usual.

## 7. Implementation note (2026-09-08)

Steps 1, 2 and 4 of the rollout above are implemented. This section records the one substantive
change made on the way: the ordering key is `ell_comps_1`, not the ellipticity magnitude that
section 4 proposed.

### 7.1 What the phase-4 runs actually contain

The ten `euclid_dr1_prelim` phase-4 tiles (RAL job 342301), maximum-likelihood lens-light
ell_comps, set A and set B as the run labelled them:

| tile | set A (e0, e1) | set B (e0, e1) | delta e1 | \|e_A\| | \|e_B\| |
|---|---|---|---|---|---|
| 102005065 | (0.007, -0.500) | (-0.023, 0.497) | -0.997 | 0.500 | 0.498 |
| 102007299 | (-0.024, 0.121) | (-0.284, 0.110) | 0.011 | 0.123 | 0.305 |
| 102007899 | (0.039, 0.497) | (0.339, -0.193) | 0.690 | 0.499 | 0.390 |
| 102007903 | (0.305, -0.342) | (0.013, 0.153) | -0.495 | 0.458 | 0.154 |
| 102008165 | (0.209, -0.045) | (-0.498, -0.428) | 0.383 | 0.214 | 0.657 |
| 102008219 | (0.001, 0.019) | (-0.500, -0.496) | 0.515 | 0.019 | 0.705 |
| 102008468 | (0.426, 0.047) | (-0.218, -0.209) | 0.256 | 0.429 | 0.302 |
| 102008475 | (0.085, 0.404) | (-0.005, -0.074) | 0.478 | 0.413 | 0.074 |
| 102008532 | (-0.060, 0.140) | (0.003, 0.394) | -0.254 | 0.152 | 0.394 |
| 102008848 | (-0.286, 0.495) | (-0.007, -0.239) | 0.734 | 0.572 | 0.239 |

Tile 102005065, the clearest case in section 1, is also the case that rules out the magnitude
key: its two sets are 0.002 apart in magnitude and 0.997 apart in `ell_comps_1`. Ordering by
magnitude would be deciding the labelling on a difference three orders of magnitude smaller than
the one the data actually determine, and would forbid whichever member of the pair the ordering
came out against.

### 7.2 Why no continuous key is exact

Any ordering key that is a continuous, antisymmetric function of the two ellipticity pairs is
blind on a set of codimension one, and the only question is where that set lies.

A linear key `w . (e_A - e_B)` for a fixed direction `w` decides the labelling by the sign of a
projection, and is blind exactly when `e_A - e_B` is perpendicular to `w`. That is a line through
the two-dimensional space of differences: a measure-zero set, but a real one, and a nearby
difference is decided by a margin that the sampler's own resolution can flip.

A rotation-invariant key cannot depend on the direction of the difference at all, so it reduces
to a function of the two magnitudes, and is blind wherever `|e_A| = |e_B|`. That is not an
exotic configuration for a two-basis MGE: when both sets pin against opposite edges of the
[-0.5, 0.5] box they land in the cross configuration `(x, -0.5)` and `(-x, +0.5)`, which is
equal in magnitude by construction. Tile 102005065 is that configuration, and it is the tile the
degeneracy was first noticed on. A rotation-invariant key is therefore blind precisely where
this model is most likely to need it.

The choice is between a blind set that lies where the data are ambiguous anyway and one that
lies where the data are sharp. `e1_A > e1_B` puts it in the first place; the magnitude puts it
in the second.

### 7.3 The key, and its blind band

The implemented key is `e1_A > e1_B`: the `cos 2phi` component of the first basis must exceed
that of the second. On the table above it separates nine of the ten tiles by more than 0.25.

The tenth, 102007299, has `delta e1 = 0.011`, inside the width the search resolves. There the
ordering does not settle the labelling, and the assertion merely picks whichever side of a
near-tie the sampler happened to land on. That is the blind band, and it is readable from the
result itself: take the two sets' `ell_comps_1` from a fit and compute `|delta e1|`. A value
comparable to the posterior width on `ell_comps_1` (order 0.05 on these tiles) means the labelling
is undetermined, and the pair must be read unordered, exactly as section 6 prescribes for
pre-ordering runs. A value well above it means the ordering is doing real work.

This is a diagnostic to run on every ordered result, not a one-off check: which tiles fall in the
band depends on the data, not on the code.

### 7.4 Where the ordering lives

The ordering is in the library, not in this pipeline:
`mge_model_from(..., order_bases=True)` (PyAutoGalaxy#610) attaches `K - 1` assertions to the
returned `Basis` model, requiring the bases' shared `ell_comps_1` values to be strictly
decreasing. It defaults to **off**, so no existing user's model or identifier changes; this
pipeline turns it on for the lens light in `vis_lp_model_from`
(`scripts/initial_lens_model.py`), together with `ell_comps_limit=0.5`.

`ell_comps_limit` is the second half of the change. The pipeline used to impose its box by
building fresh `TruncatedGaussianPrior`s after composition and reassigning them over the
returned model's Gaussians. That is incompatible with an assertion attached inside
`mge_model_from`, which references the prior objects that function built: reassigning would
leave the assertion pointing at priors the model no longer contains. Passing the box in as
`ell_comps_limit` (0.5 for the lens, 0.7 for the source) removes the reassignment loops
entirely, and the priors the assertion references are the priors the model samples.

Enforcement is backend-specific and PyAutoFit#1583 supplied the missing half. On NumPy
(`--use_cpu`) `check_assertions` raises `af.exc.FitException` and Nautilus resamples, which is
what section 4 described. Under JAX the assertions are now evaluated as a traced boolean and a
violating model is mapped to the resample figure of merit, so the vmapped `Fitness` that
Nautilus builds no longer raises `TracerBoolConversionError` at trace time. Option (a) is
therefore no longer gated on `--use_cpu`.

The search also takes `--seed` (option (c)), passed through to `af.Nautilus` for `vis_lp`. It
buys run-to-run reproducibility only, and is `None` (unseeded) by default, which is what the
witness reruns of step 3 need: two *unseeded* runs agreeing set by set is the evidence that the
ordering fixed the labelling.

Both `order_bases` and `seed` enter the PyAutoFit identifier, so every run made with them is a
fresh output directory and results predating them are untouched and not comparable set by set.

## Appendix A: demo summary

Produced by a session scratch script (not committed; it composes the model exactly as
`scripts/initial_lens_model.py:162-236` does, evaluates the analysis on the shipped example tile
with and without the sets swapped, and exercises the three `Fitness` paths). Verbatim:

```
=== SUMMARY ===
dataset                  : real shipped Euclid VIS example q1_walsmley/102018665_NEG570040238507752998, (100, 100) native, 3852 masked pixels, pixel_scales=(0.1, 0.1), mask_radius=3.5, dataset_centre=(0.0, 0.0)
prior_count before/after : 15 / 15
assertions on model      : 1
swap diff (numpy, x3)    : [0.0, 1.8189894035458565e-12, 0.0]
swap diff (jax,   x3)    : [0.0, 0.0, 0.0]
numpy  violating         : -1e+99
numpy  satisfying        : array(15581.54895082)
jax    violating         : 'RAISED FitException'
jax    satisfying        : 'RETURNED Array(15581.54895082, dtype=float64)'
jax/vmap violating       : 'RAISED TracerBoolConversionError'
jax/vmap satisfying      : 'RAISED TracerBoolConversionError'
versions                 : af=2026.8.17.1 ag=2026.8.17.1 al=2026.8.17.1
=== END SUMMARY ===
```

The run also recorded `general.test.exception_override = False`, which matters because assertions
are skipped entirely when it is True.

## Appendix B: code references

- `scripts/initial_lens_model.py:162-236` - `vis_lp` model composition; the lens MGE is 2 sets of
  20, with the +/-0.5 ell_comps prior loop at `:179-189`.
- `scripts/initial_lens_model.py:266-272` - `af.Nautilus(name="vis_lp", ..., n_live=750, ...)`, no
  `seed`.
- `scripts/initial_lens_model.py:603-604` - `vis_pix` freezes the `vis_lp` maximum-likelihood lens
  light as an instance; prose at `:559-566`.
- `PyAutoGalaxy/autogalaxy/analysis/model_util.py:7` - `mge_model_from`; the shared
  `log10_sigma_list` at `:119`, the per-basis `_make_ell_comps_priors()` at `:146-172`.
- `PyAutoFit/autofit/non_linear/fitness.py:380-394` - `Fitness.call`; the JAX branch has no
  `except exc.FitException`.
- `PyAutoFit/autofit/mapper/prior_model/abstract.py:193-226` - `check_assertions`, which raises
  `FitException`; `:1600-1628` - `instance_for_arguments`, which calls it unless
  `exception_override` or `ignore_assertions`.
- `PyAutoFit/autofit/non_linear/search/nest/nautilus/search.py:116, 242-256` - Nautilus builds
  `Fitness` with `use_jax_vmap=True` by default; `:84, 103` - `seed` as a search kwarg and
  identifier field.
- `PyAutoFit/autofit/non_linear/paths/abstract.py:274-292` - the run identifier hashes the search
  and the model together.
- `PyAutoFit/autofit/mapper/prior_model/constraint.py` - `__model_constraint__`, the traced
  per-class non-negative violation measure; consumed today by diagnostic counters, with the module
  docstring anticipating the penalty term step 1 above needs.
- `autolens_workspace/scripts/guides/modeling/cookbook.py:311-317` - the ordering-assertion
  precedent.
