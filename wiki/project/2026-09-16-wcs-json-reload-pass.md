---
date: 2026-09-16
title: pipeline main merged in and the nine completed lenses reloaded so files/wcs.json carries the source positions
tags: [euclid, dr1, dr1_sep1, initial_lens_model, wcs_json, source_clumps, reload, force_pickle_overwrite, ral]
---

# pipeline main merged in and the nine completed lenses reloaded so files/wcs.json carries the source positions

## Context

The sep1 array (`342650`) was submitted from pipeline `dbbb2a5`, whose
`util.wcs_dict_from` wrote only the lens centre — `crval_ra_deg` / `crval_dec_deg`,
`crpix_x` / `crpix_y`. Upstream has since landed four PRs on the pipeline's `main` that put
the *source* in that record: PR #71 (the lensed source's multiple images beside the lens
centre, via the `al.PointSolver` route), #72 (pixelized-source clumps read off the mapper, and
no `null` keys), #73 (a slow-suite fit proving the clumps are written and aggregator-readable),
and #79 (clumps thresholded against the reconstruction's p99 with a brightest-pixel failsafe).
The nine sep1 fits that completed therefore carry a `wcs.json` that predates every source
field — and re-fitting them to get those fields would cost 3 h+ per lens for a search whose
samples are already on disk. So this session ported the upstream work into the clone and
**reloaded** the finished searches instead of refitting them.

## What I did

**Merged the pipeline's `main` into the clone.** `git merge origin/main` at
**`650dc2d`** (the merge of PR #79) landed conflict-free as merge commit **`5fdb4de`**:
`util.py` (+485/-…), the new `tests/test_wcs_dict.py` (846 lines), `tests/pixelized_model.py`,
`tests/test_latent_run_level.py`, `AGENTS.md`, `README.md`, `catalogue/README.md`. The clone's
own two science commits (`3652592`, `046d020`) are preserved. **Nothing was pushed** — `origin`
here is the pipeline repo.

`NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python3 -m pytest
tests/test_wcs_dict.py -q -x` → **21 passed** in 8.2 s against the merged tree.

**Turned the reload lever on.** `config/general.yaml` line 30,
`output.force_pickle_overwrite: false` → `true` (commit `e5d6d83`). That flag is the whole
mechanism: with it set, a search that is already complete logs
`Fit Already Completed: skipping non-linear search` and then `Forcing pickle overwrite`, which
re-fires `AnalysisImaging.save_results` — the method that writes `files/wcs.json` — and re-zips
the result in place. No sampling happens.

**Wrote `hpc/batch_cpu/submit_reload_wcs`** (commit `b73351f`), a copy of
`submit_initial_lens_model_two_stage` reduced to one `python3` call per array task:

```
env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 python3 $PROJECT_PATH/scripts/initial_lens_model.py \
    --sample=dr1_sep1 --dataset=$dataset --stage=all --use_cpu --number_of_cores=1
```

Both stages reload in one interpreter because nothing samples: the two-process split in the
production script exists only to keep a forked Nautilus pool away from a JAX-initialised
interpreter, and `--number_of_cores=1` forks no pool. `-J euclid_dr1_reload_wcs`,
`--array=0,2-9`, `-t 00:30:00`, `--mem=8gb`, `--cpus-per-task=1`, partition `ral`, same log
paths, same dataset list in the same order so the array indices still mean the same lens.
No `--seed`, no model or search change — the identifier hashes must not move.

**Array index 1, `Tile102007299RA0702283866574DECNEG0660415308762`, was excluded.** Its
`vis_lp` never completed (`output/dr1_sep1/…/vis_lp/ddee68f8…/` has no `.completed` and no
zip), so a reload there would have started a fresh multi-hour fit rather than loading one.
Nine of the ten tiles were reloaded.

**On RAL.** 18 zips confirmed (9 tiles × 2 stages), then
`cp -a output output.prezip.bak.2026-09-16` — `paths.restore()` deletes the zip before
re-zipping, so the backup is the only thing standing between a bad reload and the fits.
`hpc/sync push --no-data` carried the merged `util.py`, the new config flag and the submit
script. Verified on RAL: `force_pickle_overwrite: true`, `SOURCE_CLUMP_SCALE_PERCENTILE = 99.0`
in `util.py`, and the installed stack (**2026.8.17.1** for autoarray / autofit / autolens)
has the API the new clump code needs — `AbstractInversion.mappings_from`, `.cls_list_from`,
`.reconstruction_dict`, and `util` imports clean.

**Ran it.** `hpc/sync submit cpu submit_reload_wcs` → **SLURM array `343358`**, all nine tasks
RUNNING immediately on `euclid-ral-compute-1`, all nine `COMPLETED` within the first poll:

| task | tile | elapsed |
|---|---|---:|
| `343358_0` | `Tile102005065RA0135279431487DECNEG0701599765928` | 00:00:34 |
| `343358_2` | `Tile102007899RA0631697483678DECNEG0650586413593` | 00:00:33 |
| `343358_3` | `Tile102007903RA0668824944366DECNEG0648906250124` | 00:00:31 |
| `343358_4` | `Tile102008165RA0109661927007DECNEG0642904268129` | 00:00:32 |
| `343358_5` | `Tile102008219RA0727857418392DECNEG0644388277074` | 00:00:33 |
| `343358_6` | `Tile102008468RA3567393529734DECNEG0647169093056` | 00:00:32 |
| `343358_7` | `Tile102008475RA0039777627054DECNEG0637715426503` | 00:00:34 |
| `343358_8` | `Tile102008532RA0683486015030DECNEG0642073552911` | 00:00:31 |
| `343358_9` | `Tile102008848RA0601371893270DECNEG0634606579714` | 00:00:29 |

Every `.out` log shows `Forcing pickle overwrite` **twice** and
`Fit Already Completed: skipping non-linear search` **twice** (once per stage); no `.err`
carries a traceback — only the workspace-version and mask-padding `UserWarning`s.

**Pulled, flattened.** This clone's `output/` is flat (`output/<Tile>/…`) where RAL's is
`output/dr1_sep1/<Tile>/…`, so the pull was an explicit rsync of
`euclid_jump:…/output/dr1_sep1/` → `output/` with the sync script's own flags, plus the SLURM
logs into `hpc/batch_cpu/{output,error}/`. `hpc/sync pull` was deliberately *not* used: it
would have recreated the nested layout beside the flat one.

Nine leftover pre-fit shadow `<hash>/` directories — a directory with no `.completed` sitting
beside a `<hash>.zip` of the same name, left by the original fits — were deleted locally, since
`restore()` had already removed them on RAL. Local `output/` fell from ~140 MB to 52 MB and now
mirrors RAL exactly: 18 zips and the one genuinely incomplete `vis_lp` directory of the
excluded tile.

**Turned the lever back off.** `force_pickle_overwrite: false` (commit `b1e0e23`, the file
byte-identical to its pre-pass state), pushed, and confirmed `false` on RAL.

## Outcome

**All 18 records now carry the source fields.** In the reloaded zips only two members changed:
`files/wcs.json` grew from **186 → 1359 bytes**, and `files/tracer.json` shrank
(104899 → 88736 bytes on `Tile102005065`'s `vis_lp`) because the current library serialises the
tracer more compactly than the one that fitted it. The 29-entry zip membership is otherwise
identical, the 18 `<hash>` identifiers are unchanged (`diff` against
`output.prezip.bak.2026-09-16` is empty), and no loose `<hash>/` directory remains.

The verification read `files/wcs.json` out of every local zip (PyAutoFit's
`type`/`arguments`/`values` envelopes unwrapped):

| tile | stage | source_model | centre (y,x)" | images | clumps | rule | brightest clump: peak(y,x)" val mesh imgs |
|---|---|---|---|---:|---:|---|---|
| `Tile102005065…0701599765928` | vis_lp | light_profile | +0.1169,−0.0372 | 4 | – | – | – |
| `Tile102005065…0701599765928` | vis_pix | pixelized | +0.1222,−0.0988 | 2 | 3 | percentile | +0.1222,−0.0988 v=1.479 n=3 imgs=2 |
| `Tile102007899…0650586413593` | vis_lp | light_profile | −0.4285,+0.2416 | 2 | – | – | – |
| `Tile102007899…0650586413593` | vis_pix | pixelized | −0.5179,+0.2085 | 4 | 4 | percentile | −0.5179,+0.2085 v=1.985 n=6 imgs=2 |
| `Tile102007903…0648906250124` | vis_lp | light_profile | −1.1822,−0.7815 | 4 | – | – | – |
| `Tile102007903…0648906250124` | vis_pix | pixelized | −0.8532,−0.6232 | 4 | 1 | percentile | −0.8532,−0.6232 v=1.344 n=17 imgs=3 |
| `Tile102008165…0642904268129` | vis_lp | light_profile | −0.5027,−0.2611 | 2 | – | – | – |
| `Tile102008165…0642904268129` | vis_pix | pixelized | −0.4697,−0.3179 | 2 | 2 | percentile | −0.4697,−0.3179 v=8.850 n=3 imgs=2 |
| `Tile102008219…0644388277074` | vis_lp | light_profile | +0.6293,−0.4470 | 2 | – | – | – |
| `Tile102008219…0644388277074` | vis_pix | pixelized | +0.6863,−0.3508 | 2 | 2 | percentile | +0.6863,−0.3508 v=0.818 n=19 imgs=3 |
| `Tile102008468…0647169093056` | vis_lp | light_profile | +0.3121,+0.1212 | 2 | – | – | – |
| `Tile102008468…0647169093056` | vis_pix | pixelized | +0.3828,+0.2325 | 2 | 2 | percentile | +0.3828,+0.2325 v=0.494 n=21 imgs=2 |
| `Tile102008475…0637715426503` | vis_lp | light_profile | +0.9604,−0.5251 | **1** | – | – | – |
| `Tile102008475…0637715426503` | vis_pix | pixelized | +1.0460,+1.2249 | 2 | 2 | percentile | +1.0460,+1.2249 v=0.199 n=3 imgs=2 |
| `Tile102008532…0642073552911` | vis_lp | light_profile | +0.1130,−1.0561 | 4 | – | – | – |
| `Tile102008532…0642073552911` | vis_pix | pixelized | +0.3390,−1.0268 | 2 | 1 | percentile | +0.3390,−1.0268 v=58.733 n=6 imgs=2 |
| `Tile102008848…0634606579714` | vis_lp | light_profile | −0.1445,−0.2901 | 2 | – | – | – |
| `Tile102008848…0634606579714` | vis_pix | pixelized | −0.1464,−0.3202 | 2 | 2 | percentile | −0.1464,−0.3202 v=20.137 n=7 imgs=2 |

Every record has a `source_model` and a `source_centre_*`; every `vis_pix` has at least one
clump with a peak and image pixels; **every clump came from the `percentile` rule** — the
brightest-pixel failsafe PR #79 added never fired on this sample, and no record is missing its
sky-coordinate keys. The mechanism is sound on all nine lenses.

**Two disagreements between the two stages** (separation of the `vis_pix` brightest-clump peak
from the `vis_lp` source centre; the rest sit at 0.03"–0.23"):

- `Tile102008475…0637715426503` — **1.75"**, and the same tile's `vis_lp` centre solves to only
  **one** image where every other record has 2–4. A single image means that source position is
  not inside the caustic: the light-profile stage put the source where it would not be multiply
  imaged, and the pixelized stage put it 1.75" away. This is a fit disagreement, not a record
  bug; it is the strongest "this tile is not reproducing" signal so far and it is the tile whose
  `mask_radius` is the largest in the sample (6.81").
- `Tile102007903…0648906250124` — **0.37"**, modest, and its `vis_pix` reconstruction is a single
  17-pixel clump.

## Next

- [ ] Fold the source positions into the by-eye sheet: `source_clumps` now gives the image-plane
      pixels of every clump, so the side-by-side against prelim `342629` can carry markers.
- [ ] Judge `Tile102008475…0637715426503` by eye — a one-image `vis_lp` solution and a 1.75"
      stage disagreement is either a failed lens or a failed `vis_lp`.
- [ ] Decide on `Tile102007299…0660415308762`, the excluded tile: its `vis_lp` never finished,
      so it has no record to regenerate and is absent from the nine.
- [ ] Delete `/mnt/ral/jnightin/euclid_dr1/output.prezip.bak.2026-09-16` (109 MB) once the
      reloaded results are trusted — it is the pre-reload copy of all 18 zips.
