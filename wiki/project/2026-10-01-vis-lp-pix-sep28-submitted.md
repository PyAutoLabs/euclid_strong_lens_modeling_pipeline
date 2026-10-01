---
date: 2026-10-01
title: vis_lp + chained vis_pix submitted for the sep28 eclipse new_ab batch (987 lenses, dr1_v2_new)
tags: [euclid, dr1, dr1_v2_new, sep28, vis_lp, vis_pix, ral, submitted]
---

# vis_lp (384040) + chained vis_pix (384046-385042) for `dataset/dr1_v2_new/`

These are the v2 A/B lenses that `catalogue/grades/dr1_v2_new_cutouts_needed.csv` asked the
lens-finding team for. The team delivered them on 2026-09-28. Every usable new tile now has
a vis_lp task queued and a vis_pix task chained to it.

## Delivery

- `sep28_eclipse_new_ab-segmentation.zip` (1.03 GB, 14,788 entries) is the data. It holds
  1,056 tile dirs, and each has the same 12-member layout:
  - `<name>.fits`, `<name>.png`, `info.json`, `positions.json` and `segmentation.png`;
  - `segmentation/{lens,source,artefact}_{flux,binary}.fits`.
  It also has a top-level `segmentation/` dir of overview PNGs, a `.DS_Store` and
  `__MACOSX/`; none of these were extracted.
- The FITS hold 25 HDUs (PRIMARY + 8 bands x BGSUB/PSF/RMS). Sizes vary with band coverage,
  from 665,280 to 987,840 B.
- No `info.json` is empty and no dir is missing a file.
- `sep28_eclipse_new_ab.zip`, the zip without `-segmentation`, has the same 1,056 FITS and
  nothing else. It was not used.
- `lens_binary.fits` is uint8 here (14,400 B), where it was int32 in sep1 (43,200 B).
  `util.load_vis_dataset` loads both kinds; it was checked locally on two tiles.

### `15arcsec_near_RADEC_`

14 dirs carry this prefix, and 12 of them are queued. The prefix is part of the v2
catalogue's own cut-out id: the same strings are `id_str_for_cutout` in
`DR1_full_Object_Catalogue_v2.csv` and `id_str` in the cut-out catalogue.

- The RA/DEC in the name belong to a nearby reference tile, 0.5-12.3" away.
- The cut-out is centred on the object itself: the WCS centre is 0.00-0.06" from
  `RA_obj`/`DEC_obj` on all 14.
- The dir name equals the FITS stem, so the pipeline fits these tiles as they are.

## Selection (`hpc/run_manifests/dr1_v2_new_selection_20261001.csv`, one row per dir)

| class | n | A | B | action |
|---|---:|---:|---:|---|
| in `dr1_v2_new_cutouts_needed.csv` | 942 | 99 | 843 | queued |
| re-centred cut-out of an object that dr1_sep1 holds via a `name_cutout` match, with no dr1_sep1 tile within 1" | 45 | 15 | 30 | queued |
| name is already a dr1_sep1 tile | 48 | 28 | 20 | excluded (duplicate) |
| re-centred cut-out within 1" of a dr1_sep1 tile | 21 | 16 | 5 | excluded (duplicate pointing) |
| broken | 0 | | | |

Notes on the selection:

- **The 45 queued re-centred cut-outs.** Each object already has a dr1_sep1 tile, matched
  by name to a cut-out (`name_cutout`). That tile was cut 1-12" off the object, so the new
  cut-out is the first one that is centred on it. The 45 include 10 of the
  `15arcsec_near_` dirs.
- **3 needed objects were not delivered:** `Tile102010485...`,
  `Tile102012697...` and `15arcsec_near_RADEC_Tile102162315...`. All three are B.
- **The 69 excluded dirs are 6.5 % of the delivery.** They are all clean duplicates: 48
  have exactly the name of a dr1_sep1 tile, and 21 are within 1" of one.
- **Queue:** 987 tiles (A 114, B 873; 950 Systematic, 37 50deg). They are ordered by
  `v2_score` descending, ties by name (2.99999 -> 1.500), in
  `hpc/run_manifests/vis_lp_sep28_20261001.txt`. Task N = line N. It is one part, so no
  split.

## Preprocessing (local, then shipped)

The sep1 tiles were preprocessed with a chain of one-off tools, and the same chain was run
here, in the same order, on `dataset/dr1_v2_new/` (987 dirs x 11 files, 1.3 GB):

1. **RGB.** For each tile, `images/NNNNN_<tile>_STCI_Euclid_10x10arcsec.jpg` was taken
   from `Euclid_DR1_v2_ABC_24078_ranked_cutout_JPG_20260929.zip` and written as
   `<tile>/rgb.jpg`.
   - 987/987 were found, and each tile matched exactly one member.
   - Member names, sizes and CRCs are in
     `hpc/run_manifests/rgb_manifest_dr1_v2_new_20261001.csv`.
   - The WebP zip is a mosaic explorer (one HTML file and 1,285 map tiles), not per-tile
     images. It was not used.
2. **Flip.** `preprocess/flip_rgb.py --dataset-root dataset/dr1_v2_new` flipped 987. The
   new JPGs carry the same vertical mirror as the 09-20 sep1 JPGs. On 7 sampled tiles the
   correlation of VIS with the flipped JPG beat the as-is JPG: 0.60-0.99 against
   0.09-0.81. The sep1 raw/fixed pairs give the same signature.
3. **Artefact fix.** `preprocess/fix_artefact_binary.py dataset/dr1_v2_new` left all 987
   unchanged and removed 0 px, because this delivery's artefact maps already exclude
   lens and source. `artefact_binary_raw.fits` exists all the same, and the CSV is
   `dataset/artefact_fix_dr1_v2_new.csv`. `redraw_segmentation_png.py` was not needed,
   since nothing differs from raw.
4. **Positions.** `scripts/tools/rewrite_positions.py --root dataset/dr1_v2_new --nproc 8`
   (finder 2.1) took 227 s with 0 errors. Results: keep 633, drop 59, flag 7, review 9,
   n_lt_2 279.
   - The 288 tiles in review or n_lt_2 have no `positions.json` and fit with no positions
     penalty, as before.
   - **28 % n_lt_2 is high.** dr1_sep1_rest was 16 % (2,283 / 14,033). The rate is high in
     both grades (A 23/114, B 256/873).

## Upload / RAL checks

- The tarball (808 MB gz) was rsynced at about 8 MB/s and its md5 matches on RAL. It was
  then extracted on the server, which was slow: about 2.5 h for 15,717 files on
  /mnt/ral NFS under load. The tarball was removed afterwards.
- All 15,717 file sizes match between local and RAL.
- md5 matches 60/60 on a 1-in-50 sample of tiles (the FITS, `rgb.jpg` and
  `positions_meta.json` of each).
- `/mnt/ral` is at 95 % (9.5 TB free). The batch adds about 1.3 GB of data and roughly
  6 GB of output (dr1_sep1_rest outputs are about 6 MB per lens), so space is not tight.
  The fill level is worth watching, though: 70 TB were free on 09-18.
- Code push:
  - `hpc/sync push --no-data` stalled after `hpc/batch_cpu/` while walking the 45k-file
    `hpc/` log tree, and timed out with rsync 255.
  - The remaining code dirs and root files went up with the same rsync flags.
  - These files have the same md5 on RAL as locally: the 3 new or changed scripts, the
    manifest, the selection and RGB CSVs, `initial_lens_model.py` (`e71b7844`),
    `util.py` and `activate.sh`.

## Jobs (submitted 2026-10-01 ~16:25)

| job | script | tasks | resources | partition |
|---|---|---|---|---|
| 384040 | `submit_initial_lens_model_vis_lp_sep28_p01` (`euclid_dr1_vis_lp_sep28_p01`) | `0-986%100` | 4 CPU / 12 GB / 18 h | ral |
| 384046 .. 385042 | `submit_initial_lens_model_vis_pix_sep28_p01` (`euclid_dr1_vis_pix_sep28_p01`) | 987 single-task arrays `--array=N --dependency=afterok:384040_N --kill-on-invalid-dep=yes` | 8 CPU / 16 GB / 24 h | ral |

- The output root is `output/dr1_v2_new/<tile>/`. It did not exist before submission.
- **ral only.** Neither script lists `gpu`. squeue shows `ral` on all 1,974 queued tasks.
- The vis_pix chain was submitted by `hpc/batch_cpu/submit_vis_pix_rest_chain.sh`, which
  is now generalised with `--batch=rest|sep28`. `rest` is the default and behaves as
  before: a part-01 dry run on RAL reported `already_have_vis_pix=945`.
  - The run log is `~/vis_pix_sep28_chain_20261001_run1.log` on RAL.
  - The id range 384046-385042 includes 10 ids from other users. Exactly 987 are ours.
- Checks right after submission:
  - vis_lp: 987 tasks `PENDING (Priority)`.
  - vis_pix: 987 `PENDING (Dependency)`, none `DependencyNeverSatisfied`.
  - `scontrol show job 384500`: `afterok:384040_453(unfulfilled)`, 8 CPU, 16G, 1-00:00:00,
    ral.
  - A second `--dry-run` reports `already_have_vis_pix=987` with nothing to submit.
- At submission the rest vis_lp (about 2,800 tasks PENDING) and the rest vis_pix chains
  were ahead of this batch on ral, so most of it will start after them.

## Check / pull

    ssh euclid_jump 'squeue -u jnightin -r -h -o "%j %T %r" | grep sep28 | sort | uniq -c'
    ssh euclid_jump 'sacct -j 384040 -X -n -o State | sort | uniq -c'
    ssh euclid_jump 'bash /mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu/submit_vis_pix_rest_chain.sh --batch=sep28 --dry-run'   # stragglers
    hpc/sync pull output

## Next

- Once vis_lp is done, re-run the chain driver (`--batch=sep28`) to catch race casualties.
- Then run the SED chain for `dr1_v2_new`.
- The 3 undelivered needed objects go back to the lens-finding team.
