---
date: 2026-09-18
title: Bare-field pipeline deployed and top-1000 vis_lp array submitted
tags: [euclid, dr1, dr1_sep1_top1000, fields, vis_lp, ral, submitted]
---

# Bare-field deployment and top-1000 submission

## Context

The user approved the upstream contour repair, green-CI merges and science
deployment, then explicitly lifted the modelling-script submission hold:
“ah you are right, then let us submit dont worry about the comment”.
The approved model remains `fields=field`; its shear paths are
`model.fields.shear.gamma_1/2`. No collection wrapper or extra model comment was added.

## Deployment

- Pipeline PR #90 merged at `9cdee7b10e916b4e2d172257f26c37881f094802` after all nine CI checks passed.
- PyAutoGalaxy #623 merged at `90e757d336e62b75790befacc77e2d6dde460879`; integration coverage is autogalaxy_workspace_test #123 at `ae45e490f75260533d0f18bb3bc2ce951377a0d5`.
- Local science code moved from `d53b9ce2b7a34b518f659adef2629441241521b4` to `2430e4ecd0815ee1cfb4809e3836e178dd5a90b6`, selectively porting 21 reviewed files and preserving the 18 earlier science commits. This clone was never pushed to its public pipeline origin.
- RAL project `/mnt/ral/jnightin/euclid_dr1` has no Git metadata. The queue was empty before the inspected `HPCPullPyAuto` function refreshed the shared source checkouts.

| RAL library | Before | After |
|---|---|---|
| PyAutoNerves | `8eca4b3e8cdf72ed4c13850ed5b333ec7306524d` | unchanged |
| PyAutoFit | `7c0e79aab79bb7a238d0654c40a883eff0333ec3` | unchanged |
| PyAutoArray | `192d4b70215830ad3ad3c8c83550e477bd674b72` | unchanged |
| PyAutoGalaxy | `33714b800b9239ada5e0531822883cf302703e1d` | `90e757d336e62b75790befacc77e2d6dde460879` |
| PyAutoLens | `7197380671c0a32be1a225d4de660b9e4097f6c7` | `478213e787781517113e96f80013270df482f665` |

All tracked shared-library trees were clean before submission. The three editor
backup files in Array, Galaxy and Lens retained their exact hashes.

Code/configuration were deployed through `hpc/sync`. Its `push` dispatcher ignores
`--dry-run`, so an attempted preview began the already-authorized sync. The remote
code/config snapshot overlapped the first transfer directories and is not an
atomic pre-deployment snapshot. Existing science outputs were preserved.

The slow network transfer was completed with a resumable lossless archive, then
extracted in disjoint batches. The final archive SHA-256 is
`c1bf8d6287aa0d3b6f4f2b7e05165d08b759a005d8a4b472b1b0bfd2f98e392a`.
The complete extraction replaced provisional partial uploads before verification.

## Validation

- Original source gates: 230 pipeline tests, 9/9 smoke scripts, independent review CLEAN; upstream repair covered JAX 0.10.2 and 0.11.2, 1236 library tests and the previously failing Euclid latent trace.
- Actual RAL interpreter `/mnt/ral/jnightin/PyAuto/PyAuto/bin/python3` imported the canonical shared source trees. The representative model has 15 priors, flat shear paths, bare `MassField`, list-valued `tracer.fields`, and `CompatibleZeroSolver`.
- The reviewed contour regression passed on RAL's JAX 0.10.2: values, JIT, bounded vmap, gradients and both LensCalc construction paths.
- The first selected real dataset passed the approved script locally in `PYAUTO_TEST_MODE=2`: one likelihood evaluation, sampler bypassed, output isolated under `/tmp/euclid-flat-fields-top1000-preflight-20260918`.
- Exactly 1,000 unique CSV identifiers match the submit script's order. All 11,000 prepared input files and 115 deployed code/configuration hashes matched before submission, with zero mismatches.
- CSV SHA-256: `4a5828de1569ebf9e588addb28b32c0dcf1f433650a89a6c660402eff3391eb4`.
- Submit-script SHA-256: `e680fcee883a0276eb2405a7c54ff64f8d3f830936b3261a34a2a36e5362bd15`.

Physical likelihoods agree with the former model, but parameter paths change the
model identifiers. Historical fit directories were preserved, not rewritten or
treated as resumable under the new identifiers. Release gates remain separate;
this was source-checkout deployment, not a package release.

## Submission

```bash
hpc/sync submit cpu submit_initial_lens_model_vis_lp_top1000
```

SLURM accepted **array `343480`**, indices `0-999`, partition `ral`, 8 CPUs,
64 GB, 18 hours per task. It runs `scripts/initial_lens_model.py --stage=vis_lp`
on `dr1_sep1_top1000`, with JAX on CPU. The initial queue showed 154 running
and 846 pending tasks. No `vis_pix` follow-up was submitted.

Results belong under `output/dr1_sep1_top1000/<tile>/initial_lens_model/vis_lp/`.
Submission and running state were recorded immediately with the Cortex verbs.
This is a launch record, not a scientific result or convergence judgment.

## Next

- Check array `343480` through `hpc/sync jobs` / `sacct`, and pull outputs when appropriate.
- Inspect completed fits before choosing any further stage; no later submission is implied.
