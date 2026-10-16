"""
Euclid Pipeline: CSV-Only Catalogue Snapshot
============================================

Metadata-only catalogue tables read out of the stored result zips, with no
refitting and no image extraction. Promoted from the ad-hoc 2026-10-11 build
(``inspect_download/csv15067_build_20261011/``) into a tracked, parameterised
tool; the table set and column semantics are those of the 2026-10-11
``dr1_catalogue_15067_20261011_csv_only`` release so later snapshots diff
against it.

Modules (run ``build.py``; the others are its stages):

- ``common.py``   shared paths, manifest / object-catalogue readers, cutoff parsing.
- ``collect.py``  one JSON line per manifest tile: the newest completed result per
  stage / band at or before the cutoff (summary, latent summary, wcs.json).
- ``assemble.py`` the CSV tables from the collected lines.
- ``package.py``  master_columns.csv, README.txt, validation.csv, file_manifest.csv.
- ``build.py``    the single entry point: collect -> assemble -> package.
"""
