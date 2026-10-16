# 2026-10-16 — v1.1 release chain submitted

Human steer this session: no more fixing of hanging or odd lenses; wrap up the core
work (catalogue, clean-up). One last pass of `output_sed` is fine.

## Rerun outcomes (10-15 submissions)

- **vis_lp 410758 (105 tiles):** 64 COMPLETED, 41 TIMEOUT at 18 h. All 41 logged
  their last line at 23:38-23:39 (right after the vmap JIT) and sat at Nautilus
  `Calls 0` for 18 h: the known deterministic hang (2026-09-30 notes: 45/46 re-hang).
  Spread over both nodes at the same rate, so not node-specific. Given up: these
  tiles are excluded from v1.1. A 30-min py-spy diagnostic (411979, scratch
  `hang_diag_20261016/`) was submitted before the steer; its dumps are unread.
- **vis_pix 410759 / SED 410760** started together (both `afterany` on 410758),
  so the 41 hung tiles fail in both with "expected one completed vis_lp zip/seed".
- **Race:** a vis_pix task and an SED task that open the same tile's vis_lp result
  at the same moment collide. PyAutoFit unzips, prunes and re-zips the result
  folder, and the reader can see it half-pruned (`'NoneType' object has no
  attribute 'galaxies'`, e.g. 410760_6 on Tile102007843…). The zips themselves are
  intact: 474 of 474 vis_lp zips over the two manifests hold `files/model.json`.
  3 of the first 16 SED failures were race victims. Accepted as best-effort; never
  run two arrays that load the same result concurrently again.

## Release chain (all `ral`)

- Packager synced to RAL (`package_release.py`, `release_util.py`, `csv_snapshot/`,
  `merge_catalogue.py`, `submit_release_package`, `submit_catalogue_chunked.sh`).
  RAL's previous `merge_catalogue.py` and `submit_catalogue_chunked.sh` are kept as
  `*.bak_20261016`; the local `merge_catalogue.py` keeps the h5py `RuntimeError` fix
  (helpers moved into `release_util.py`).
- Dry run: 15,067 manifest tiles, 6 chunks of ≤2,512, 13:53 walltime each.
- **Gate 412011** `hpc/batch_cpu/submit_v1_1_release_chain`, `afterany:410759:410760`,
  `CATALOGUE=dr1_catalogue_v1_1`. It writes
  `hpc/run_manifests/v1_1_excluded_dr1_catalogue_v1_1.csv` (manifest tiles with no
  vis_lp zip), runs the chunked build (its `tiles.txt` mtime = snapshot cutoff, now
  after the reruns), then `submit_release_package` `PARTS=10` afterok on the merge.
  Job ids land in `hpc/run_manifests/v1_1_chain_dr1_catalogue_v1_1.jobs`.
- Code provenance: the shared `/mnt/ral/jnightin/PyAuto` HEADs equal the pinned 09-27
  SHAs in `PyAuto_dr1/PINNED_SHAS.txt` (Lens 21b520be4, Fit 404b3e5f7, Array
  9428eca2, Galaxy c9609825, Nerves bf10410). The frozen venv `PyAuto_dr1` itself
  FAILED (pip ResolutionImpossible on `numpy==2.2.6`); not needed for the build.

## Next

1. Read `v1_1_chain_*.jobs`; check chunk/merge logs under `inspect/dr1_catalogue_v1_1/_build/logs/`.
2. If the packager stops on absent tiles: review
   `inspect/dr1_catalogue_v1_1_release_absent_tiles.csv`, re-submit
   `submit_release_package` with `EXCLUDED=<that csv> SKIP_CSV=1`.
3. Verify each tier (`sha256sum -c SHA256SUMS`), then pull/upload.
4. After v1.1 is verified: archive the v1.0 dist (102 GB) and `dr1_catalogue_20261005*`.
