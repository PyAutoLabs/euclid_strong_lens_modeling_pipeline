<!-- Provenance: this file is GENERATED, not hand-maintained. -->

> **2026-08-29 — the table below is stale and was not regenerated here.**
>
> `test_report.md` is written by PyAutoHands' `run_all.py`, which walks the
> workspaces under the canonical checkout and copies each run's markdown summary
> into the workspace root. It is not runnable from a task worktree, so this file
> is refreshed by the next release/CI run rather than by hand.
>
> Two things to know when reading the table until then:
>
> - The `full_model.py` row **predates the Delaunay switch**. That PASS was
>   recorded against a version of the script that could no longer run at all:
>   `al.mesh.RectangularAdaptImage` had been removed from the library. Both
>   Source Pix stages now use `al.mesh.Delaunay` + `al.reg.AdaptSplit`
>   (`docs/drift_report.md` §4).
> - The table covers 5 scripts. `smoke_tests.txt` now lists 8.
>
> Local evidence, 2026-08-29, on branch `feature/euclid-pipeline-parity`, run
> with the same recipe the smoke runner uses (`build_env_for_script` over
> `config/build/profile_smoke.yaml` + `no_run.yaml`, cwd = repository root,
> `PYAUTO_TEST_MODE=2`) — **8/8 passed**:
>
> | Script | Time |
> |---|---|
> | `start_here.py` | 18.8s |
> | `scripts/initial_lens_model.py` | 16.4s |
> | `scripts/full_model.py` | 26.9s |
> | `scripts/lens_model_waveband.py` | 42.0s |
> | `scripts/mge_lens_only.py` | 39.5s |
> | `scripts/sersic_lens_model.py` | 18.4s |
> | `scripts/sersic_lens_model_waveband.py` | 30.1s |
> | `scripts/diagnose_latent.py` | 7.6s |

---

# Test Report: euclid / scripts (script)

**5 scripts** | 5 passed

| Status | Count |
|--------|-------|
| passed | 5 |

## Passed

- `/home/jammy/Code/PyAutoLabs/euclid_strong_lens_modeling_pipeline/scripts/full_model.py` (32.6s)
- `/home/jammy/Code/PyAutoLabs/euclid_strong_lens_modeling_pipeline/scripts/initial_lens_model.py` (19.7s)
- `/home/jammy/Code/PyAutoLabs/euclid_strong_lens_modeling_pipeline/scripts/lens_model_waveband.py` (69.4s)
- `/home/jammy/Code/PyAutoLabs/euclid_strong_lens_modeling_pipeline/scripts/mge_lens_only.py` (50.4s)
- `/home/jammy/Code/PyAutoLabs/euclid_strong_lens_modeling_pipeline/scripts/sersic_lens_model.py` (30.7s)
