"""
Euclid Pipeline: Start Here
===========================

Welcome. This file is the front door to the Euclid strong lens modeling
pipeline, and it is written to be read from top to bottom by someone who has
never fitted a strong lens before.

It has two jobs. The first is functional: ``python start_here.py --dataset=...``
runs a complete automated lens model, because the code at the bottom of this
file forwards straight into ``scripts/initial_lens_model.py``. The second is
educational: everything between here and that code explains what the pipeline
does, what it needs from you, what it writes out, and which concepts you need in
order to trust the numbers it produces.

You can stop reading at any point. If you only want results, the two sections
"__Installation__" and "__Running The Pipeline__" are enough — the pipeline is
designed to run as a black box, and the output it writes to ``output/`` is
meant to be inspected without knowing how any of it was computed. If you want to
understand the model, keep going: the later sections build up the science one
piece at a time.

__What This Repository Is__

This repository is a packaged, runnable version of the Euclid strong lens
modeling pipeline. It uses **PyAutoLens** to perform automated strong lens
modeling of Euclid VIS cut-outs — the small postage-stamp images the Euclid MER
pipeline produces around each lens candidate.

Given one cut-out, the pipeline fits a model of the foreground lens galaxy's
light, a model of its mass, and a reconstruction of the lensed background
source. From that fit it derives the quantities strong lensing science actually
consumes: an Einstein radius, a magnification, deblended images of the lens and
of the lensed source, and calibrated fluxes for both.

It is deliberately a *pipeline* and not a library. Every decision that a lens
modeller would normally make by hand — where to centre the mask, how many
Gaussians to use, when to switch from a smooth source to a pixelized one, when
to stop sampling — is already made, tuned on Euclid Q1 and DR1 data. That is
what makes it possible to point it at a thousand cut-outs and walk away.

__What This File Is__

``start_here.py`` is a **thin shim**. The real implementation lives in
``scripts/initial_lens_model.py``, and that is the file to read when you want to
see the model being built line by line, or to edit when you want to change it.
The README, ``AGENTS.md`` and the HPC submit scripts all name that script.

This file exists for two reasons. It keeps older commands, bookmarks and job
scripts that call ``python start_here.py`` working unchanged, and it is the
place where the concepts are explained once, so that the scripts themselves can
stay close to the code and point back here.

There is no duplicated fitting logic below. ``fit`` is imported from
``scripts/initial_lens_model.py`` and re-exported, so the two entry points can
never drift apart. (They once did: this file used to be a diverged copy that had
fallen a whole pixelized-source stage behind. Collapsing it into a shim removed
that failure mode permanently.)

__Installation__

**PyAutoLens** supports Python 3.12 and later, with Python 3.13 recommended. You
may want a virtual environment or conda environment to install into first.

::

    pip install --upgrade pip
    pip install autolens

Then clone this repository and change into it:

::

    git clone https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline
    cd euclid_strong_lens_modeling_pipeline

That is the whole installation for a CPU run. `README.md
<https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline>`_ has the
full instructions, including the GPU setup described next.

__JAX And GPUs__

**PyAutoLens** runs significantly faster on GPUs — often **50x or more**
compared to CPUs. That acceleration comes from
`JAX <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_, which
provides GPU and TPU support.

Installing **PyAutoLens** installs JAX as a dependency, but the default JAX
install may not include GPU support. If you have a GPU, install JAX with GPU
support **first**, following the
`JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_,
and install **PyAutoLens** afterwards. If **PyAutoLens** is installed without a
working GPU setup a warning is printed; the pipeline still runs, just on CPU.

For scale: ``scripts/initial_lens_model.py`` fits a lens in around 10 minutes on
a GPU, and around 20 minutes on an 8-core CPU (times from the DR1 science runs
under their own ``config/``; ``hpc/README.md`` has the times measured with the
committed one). Over a sample of a few thousand
candidates that difference decides whether the run is a coffee break or a
cluster allocation.

For SLURM clusters, ``hpc/README.md`` explains how to choose between the
one-job GPU route and the two-stage CPU route by sample size, and holds the
submission scripts.

__Running The Pipeline__

Every script in this repository is run from the repository root, and they all
share one argument parser (``util.parse_fit_args``). The minimal command is:

::

    python start_here.py --dataset=<name> --sample=<sample>

The repository ships a real Euclid Q1 cut-out, so this command works
immediately after cloning:

::

    python start_here.py --sample=q1_walsmley \\
        --dataset=102018665_NEG570040238507752998 \\
        --iterations_per_quick_update=10000

That is the example command from ``README.md``, with ``start_here.py``
substituted for ``scripts/initial_lens_model.py``; the two are interchangeable.

The arguments, all of which this file forwards unchanged:

- ``--dataset`` (**required**) — the dataset subdirectory inside
  ``dataset/<sample>/``.
- ``--sample`` (default: none) — the sample subdirectory inside ``dataset/``.
  Omit it for a flat ``dataset/<name>/`` layout.
- ``--iterations_per_quick_update`` (default: ``5000``) — sampler iterations
  between on-the-fly visualisation updates.
- ``--number_of_cores`` (default: ``1``) — CPU cores for the non-JAX Nautilus
  searches.
- ``--use_cpu`` (default: off) — CPU mode: disables JAX and applies the CPU
  sparse operator to the pixelized stage.
- ``--stage`` (default: ``all``) — which of the two searches to run. ``all``
  runs ``vis_lp`` and then ``vis_pix`` in one process; ``vis_lp`` returns after
  the light-profile fit, skipping the pixelized source stage; ``vis_pix`` runs
  only the pixelized stage, loading the ``vis_lp`` result from disk and failing
  immediately if it is not there. ``--skip_pix`` is still accepted as a
  deprecated spelling of ``--stage vis_lp``.

Note what is *not* on that list: ``mask_radius``. It is not a command-line
argument. It is always read from the dataset's ``info.json``, so that the mask
travels with the data rather than with the command, and a re-run of the same
dataset is always masked identically.

Two environment variables change where results go. ``PYAUTO_OUTPUT_DIR`` sets
the results directory relative to the repository root (default ``output``), and
``PYAUTO_TEST_MODE=1`` makes every search finish almost instantly with trivial
samples so you can check that a script runs before committing to a real fit.

__Watching It Fit__

The fit is not silent and it is not a black hole. Nautilus, the sampler, writes
a "quick update" every ``--iterations_per_quick_update`` iterations: the current
maximum-likelihood model, its parameter estimates, and a set of figures showing
the model image, the residuals, the deblended lens and source, and the source
reconstruction.

Those figures are written into the search's ``image/`` folder while the search
is still running. Opening that folder part-way through a fit and refreshing it
is the intended way to watch a lens model converge. Setting
``--iterations_per_quick_update`` higher (the example above uses ``10000``)
makes the fit spend less time plotting and more time sampling.

__The Dataset Contract__

A dataset is a directory. That directory is:

::

    dataset/<sample>/<name>/
        <name>.fits    # the multi-HDU cut-out
        info.json      # pixel_scale, mask_radius, optionally mask_centre
        ...            # everything else is optional

Two files are required.

``<name>.fits`` is the multi-extension cut-out: a PRIMARY header followed by a
repeating ``(<BAND>_BGSUB, <BAND>_PSF, <BAND>_RMS)`` triplet for each band, in
that order. The pipeline locates the VIS band by scanning the HDU names for the
``_BGSUB`` tag, falling back to ``_FLUX`` if a dataset was stamped with the
other convention, and derives the image, PSF and noise-map HDU indices from the
band's position in that list. You never supply an HDU index yourself.

``info.json`` supplies ``pixel_scale`` (0.1 arcsec/pixel for Euclid VIS) and
``mask_radius``, the radius in arcsec of the circular analysis mask. It may also
supply ``mask_centre``, used when the lens is not at the centre of the frame.

The mask radius is worth dwelling on, because it does more than crop the image.
It sets the outer extent of the Gaussians used to model the galaxy light (see
"__Multi Gaussian Expansion__" below), so it is a modeling choice, not just a
display choice. It is also *clamped*: if an offset lens plus a generous radius
would push the circular mask off the edge of the cut-out, the pipeline shrinks
it to the largest radius that still fits.

``dataset/README.md`` lists every dataset this repository ships, what each one
is, what reads it, and how to regenerate it.

__Optional Inputs, And What Happens Without Them__

Everything beyond those two files is optional, and every optional input degrades
gracefully — the pipeline falls back rather than failing.

- ``segmentation/lens_flux.fits`` — a map of the deblended lens light. Its peak
  is the most reliable estimate of the lens centre, so the mask is centred
  there. Without it the pipeline uses ``mask_centre`` from ``info.json``, and
  without that, the centre of the frame.
- ``segmentation/artefact_binary.fits``, or ``mask_extra_galaxies.fits`` — a
  binary map of neighbouring galaxies and artefacts. Where it is set, the noise
  is scaled up so those pixels stop pulling on the fit. The first is what the
  DR1 preprocessing writes; the second is the older convention. Whichever is
  found first is used, and a mask cut out at a different size to the image is
  ignored rather than misapplied.
- ``positions.json`` — the sky positions of the source's multiple images. These
  become a likelihood penalty that rejects mass models which cannot reproduce
  the observed image configuration, which is the single most effective guard
  against unphysical pixelized-source solutions. Without the file, the pipeline
  derives positions from ``segmentation/source_flux.fits`` if that map is
  present. Without either, it fits with no positional constraint. (A file
  holding a single position is treated as absent: one image is not a
  multiple-image constraint.)
- ``rgb_0`` / ``rgb_1`` thumbnails (``.png``, ``.jpg`` or ``.jpeg``) — colour
  composites, plotted alongside the fit for visual inspection.

The shipped Q1 dataset deliberately ships *none* of the segmentation inputs, and
the shipped simulated dataset ships all of them, so between them the two
datasets exercise both the fallback and the preferred branch of every chain
above.

__The FITS Header Contract__

Two things in the FITS **primary** header are read but never written by this
pipeline, which makes them an input contract on the cut-out generator.

``WORST_BAND`` names the worst-seeing band across all MER bands. Lower-cased, it
indexes the HDU list to find that band's PSF. ``WORST_PSF_MER``,
``WORST_PSF_HDR`` and ``WORST_PSF`` hold that PSF's FWHM in arcsec, read in that
order, skipping Euclid's ``-99`` "not measured" sentinel.

Together they enable matched-aperture photometry: the lens image is convolved to
the resolution of the worst band, and fluxes are measured at 1, 2, 3 and 4 times
that band's FWHM. If ``WORST_BAND`` is missing, or names a band absent from the
cut-out, a warning is printed and those four aperture measurements are skipped —
the lens model itself is unaffected. If ``WORST_BAND`` is present but all three
FWHM keys are missing, the load **raises**, on purpose: the aperture radii are
multiples of that FWHM, so a guessed value would silently corrupt the photometry
instead of failing loudly.

The image HDU's own header supplies ``MAGZERO``, the photometric zero-point that
converts fitted fluxes into AB magnitudes and microJansky, and the WCS that
converts the fitted lens centre into sky coordinates.

__How A Dataset Is Loaded__

All of the above is done in one call, ``util.load_vis_dataset``, which every
fitting script in this repository uses so that they all see an identically
prepared dataset. Read that function in ``util.py`` when you want the detail;
in order, it:

1. Resolves the dataset directory, ``dataset/<sample>/<name>/``, and the FITS
   file inside it.
2. Maps every waveband in the multi-extension FITS to its HDU index, by the
   ``_BGSUB`` (or ``_FLUX``) image tag.
3. Loads the VIS image, noise-map and PSF from the HDU triplet, at the pixel
   scale given by ``info.json``.
4. Chooses the mask centre — segmentation lens-flux peak, else ``mask_centre``,
   else the frame centre — and finds the brightest sub-pixel coordinate within
   0.3 arcsec of it. That coordinate anchors the model priors.
5. Reads the image header for ``MAGZERO`` and the celestial WCS.
6. Applies noise scaling from the artefact or extra-galaxies mask, if present.
7. Applies the circular analysis mask, at the ``info.json`` radius, clamped to
   fit inside the cut-out.
8. Applies adaptive over-sampling: 4x sub-pixels within 0.1 arcsec of the lens
   centre, 2x out to 0.3 arcsec, 1x beyond. Over-sampling matters most where the
   light profile changes fastest, which is the very centre.
9. Loads the worst-seeing band's PSF and FWHM for the aperture photometry.
10. Builds the multiple-image positions constraint, from ``positions.json`` or
    from the segmentation source-flux map.

It returns a single ``EuclidDataset`` dataclass carrying the masked,
over-sampled dataset plus every derived quantity — centre, zero-point, WCS,
PSF, mask radius, positions — so that a pipeline reads attributes rather than
re-deriving values and risking an inconsistency between scripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import util
from scripts.initial_lens_model import fit

__all__ = ["fit"]


"""
__Multi Gaussian Expansion__

Here is the first idea the pipeline is built on.

A galaxy's light has to be described by some mathematical function before it can
be fitted. The traditional choice is a Sersic profile — a smooth, elegant, two-
or three-parameter shape. It is also a straitjacket: real galaxies have
isophotal twists, colour gradients, boxy or discy isophotes, and a Sersic
profile forced onto them leaves structured residuals that the mass model then
tries to absorb.

A **Multi Gaussian Expansion (MGE)** takes the opposite approach. Instead of one
flexible shape, it uses a sum of many rigid ones: the galaxy is decomposed into
a basis of concentric elliptical Gaussians, typically 15 to 100 of them. Their
widths are *fixed* to a set of logarithmically spaced values running from a
fraction of a pixel out to the mask radius, so between them they can represent
structure at every scale the image resolves. Only their shared centre and
elliptical components are free.

This pipeline uses 40 Gaussians for the lens light — two independent sets of 20,
each set with its own ellipticity, so the model can capture a galaxy whose
isophotes change shape with radius — and 20 for the source.

__Why The MGE Makes This Pipeline Possible__

The MGE would be an absurd model if all of those Gaussians had to be sampled.
Sixty Gaussians with a free intensity each would be a sixty-dimensional
non-linear problem, and no sampler would find its global maximum reliably.

They are not sampled. The intensity of every Gaussian enters the model
**linearly**, and a linear problem has an exact solution. At every likelihood
evaluation, the pipeline solves for the intensities that best fit the data by
linear algebra — an "inversion" — given the current non-linear parameters. The
sampler never sees them.

The consequences are what the whole pipeline rests on:

- **The parameter space is tiny.** The initial fit has roughly 15 non-linear
  parameters in total: the lens light's centre and ellipticities, the mass
  model, and the source's centre and ellipticity. Everything that would have
  made it large is solved rather than searched.
- **The fit is robust.** A 15-dimensional space with a well-conditioned
  likelihood is one a nested sampler explores exhaustively, so the reported
  maximum is very likely the global one rather than a local trap.
- **It suits a GPU.** The expensive inner operation becomes a dense linear
  solve, which is exactly the kind of arithmetic JAX compiles and a GPU executes
  in bulk.
- **It is flexible where it needs to be.** The intensities are free to take
  whatever profile the data prefers, including profiles no Sersic index can
  produce.

For a standalone tutorial on the MGE, outside the Euclid context, see
``scripts/imaging/features/multi_gaussian_expansion/modeling.py`` in the
`autolens_workspace <https://github.com/PyAutoLabs/autolens_workspace>`_.

__The Mass Model__

Light tells you where the stars are. The lens model needs to know where the
*mass* is, and the two are related but not identical — most of a lens galaxy's
mass inside its Einstein radius is dark matter.

The pipeline models the total mass distribution as a **Singular Isothermal
Ellipsoid (SIE)** plus an **external shear**.

The SIE is the standard workhorse of galaxy-scale strong lensing. "Isothermal"
means its density falls as the inverse square of radius, which is what both
stellar dynamics and lensing measurements find for elliptical galaxies over the
radii that strong lensing probes; "singular" means it is not softened at the
centre; "ellipsoid" means it is allowed to be flattened rather than circular. It
has an Einstein radius, an ellipticity and a centre.

The external shear is a two-parameter correction for everything *outside* the
model: a neighbouring galaxy, the group or cluster the lens sits in, structure
along the line of sight. Fitting a lens without shear pushes that distortion
into the lens's own ellipticity, which biases it.

For the initial fit, the mass centre is **fixed to the brightest pixel** of the
cut-out — the same coordinate the light model is anchored to. This is a strong
assumption, and a deliberate one. The mass centre and the source position are
close to degenerate early in a fit, and pinning the mass to the light removes
that degeneracy at the moment when the sampler is least equipped to resolve it.
The assumption is relaxed later: the pixelized stage frees the centre within a
0.1 arcsec box around it, and ``scripts/full_model.py`` frees it fully.

__The Two Searches__

``scripts/initial_lens_model.py`` runs two searches, in order. Each has a name,
and those names are the folders you will find under ``output/``.

**``vis_lp``** — the light-profile fit. An MGE lens light, an SIE + shear mass,
and an MGE source, all fitted together. This is the fast, robust stage: ~15
non-linear parameters, a mass centre fixed to the light, no pixelization to go
wrong. Its job is not to produce the final answer but to produce a *reliable*
one — a mass model good enough that the next stage can be trusted to start from
it.

**``vis_pix``** — the pixelized-source fit. The lens light is now fixed to the
``vis_lp`` result, the mass centre is freed within a small box around the
brightest pixel, and the source is replaced with a pixelized reconstruction on a
Delaunay mesh. Because this stage starts from a converged mass model, its
positional constraint can be tightened using the ``vis_lp`` result, which is
what keeps the reconstruction physical.

``--stage vis_lp`` stops after ``vis_lp`` and returns its result. That is not just
a speed switch: ``scripts/sersic_lens_model.py`` uses it because its Sersic source
prior is seeded from ``galaxies.source.bulge``, and the pixelized fit replaces
that bulge with a pixelization, leaving nothing to seed from.

``--stage vis_pix`` is the other half of that split: it runs only the pixelized
stage, loading the completed ``vis_lp`` result from ``output/`` rather than
re-fitting it, and refusing to start if that result is not there. It exists for
the CPU route, where ``vis_pix`` wants a multiprocessing pool. The two stages run as separate Python processes. This is a conservative default: PyAutoFit documents an XLA deadlock for a forked worker whose likelihood touches JAX, and the DR1 science runs were submitted this way. A local control test (hpc/diagnostics/jax_fork_control.py) did not reproduce a hang for the CPU route, whose vis_pix likelihood is Numba; production sampler sizes and large pools are untested, so the boundary is kept until a cluster run passes. hpc/README.md has the measured table. On GPU, leave ``--stage`` at its
``all`` default and both searches run in one go. ``--skip_pix`` remains a
deprecated spelling of ``--stage vis_lp``.

__Pixelized Sources__

The second search is where the pipeline earns its accuracy, so it is worth
understanding the concept even if you never read its implementation.

An MGE source is a sum of concentric Gaussians. That is a good description of a
smooth, single-component galaxy and a poor one of what high-redshift sources
usually look like: clumpy, asymmetric, multi-component star-forming systems. If
the source model cannot represent the real source, the fit compensates by
distorting the *mass* model — and the mass model is the thing you wanted to
measure.

A **pixelization** removes that constraint. Instead of assuming a functional
form, the source plane is divided into pixels and the brightness of each pixel
is solved for directly. The source is then whatever the data says it is.

Two refinements make this work in practice. First, the pixels are not a regular
grid: they form an **irregular Delaunay mesh** whose points are distributed
according to the source's own reconstructed light, so the mesh is dense where
the source is bright and sparse where it is faint. A regular grid would waste
most of its pixels on empty sky and under-resolve the one clump you care about.
Second, the solution is **regularized** — neighbouring source pixels are
penalised for differing wildly — because an unconstrained pixel-by-pixel fit
would happily reproduce the noise.

The pixelized source is also what makes complex mass models possible at all. A
``PowerLaw`` or a decomposed stars-plus-dark-matter model has enough freedom to
mimic source structure; only a source free enough to absorb its own structure
lets the mass model be measured cleanly.

For the mechanics — how the mesh points are placed, how the adapt images are
built, how the mesh edge is handled — read the ``__Source Pix__`` section of
``scripts/initial_lens_model.py``, which is the implementation. A standalone
tutorial lives at ``scripts/imaging/features/pixelization/delaunay.py`` in the
autolens_workspace.

__Nautilus, JAX And The CPU Fallback__

Both searches are run with **Nautilus**, a nested sampling algorithm.

Nested sampling is not an optimizer. It maps the whole posterior rather than
walking downhill to one point, which is what makes it suitable for automation: it
returns uncertainties along with the maximum, it does not need a starting guess,
and it is far harder to trap in a local maximum than a gradient method. The cost
is many likelihood evaluations, which is precisely why the MGE's small parameter
space matters so much.

The two searches are tuned differently, because they are doing different jobs.
``vis_lp`` samples 750 live points with a large batch size, exploring an easy
space thoroughly and in parallel. ``vis_pix`` samples 300, with a smaller batch,
because each of its evaluations costs far more. Both carry a maximum-likelihood
cap so that a pathological lens cannot run forever.

By default the likelihood is evaluated through **JAX**, which just-in-time
compiles it and dispatches it to a GPU if one is available. That is where the
50x comes from.

``--use_cpu`` turns JAX off and takes the CPU path. This is not simply "the same
thing, slower": the pixelized stage applies a **CPU sparse operator**, a Numba
precomputation of the PSF products the inversion needs, because that is the
fastest way to do this linear algebra on a CPU. It is the CPU route's tool and is
applied only under ``--use_cpu`` — the JAX path applies no sparse operator and
fits the plain dataset. When you pass ``--use_cpu``, pass
``--number_of_cores`` as well — that is the argument the ``vis_pix`` search uses
to parallelise across CPU cores.

__Reading The Output__

Results are written under ``output/``, in a path built from the sample, the
dataset, the pipeline and the search:

::

    output/<sample>/<dataset>/initial_lens_model/vis_lp/<hash>/
    output/<sample>/<dataset>/initial_lens_model/vis_pix/<hash>/

The ``<sample>/<dataset>`` part is the path prefix, ``initial_lens_model`` is
the pipeline's unique tag, and ``vis_lp`` / ``vis_pix`` are the search names.
Every other pipeline in ``scripts/`` follows the same scheme with its own tag,
so results from different pipelines on the same lens sit side by side. The
``<hash>`` folder identifies a particular model and configuration; re-running an
identical fit reuses it, while changing the model creates a new one. Setting
``PYAUTO_OUTPUT_DIR`` moves the whole tree.

Inside a result folder:

- ``image/`` holds the on-the-fly visualisation, refreshed throughout the fit —
  the fit subplot, the deblended lens and source, the source reconstruction, the
  image with the multiple-image positions overlaid, and an RGB subplot built
  from the dataset's colour thumbnails by this pipeline's custom visualiser.
- ``files/`` holds the machine-readable results: ``model.json``,
  ``samples_summary.json``, and ``wcs.json`` — the last written by this
  pipeline's ``util.AnalysisImaging``, recording the fitted lens centre in sky
  coordinates so a result can be matched back to a catalogue position.

For a handful of lenses, browsing that tree directly is enough. For more than a
handful, see "__Workflow__" below.

__Latent Variables__

The parameters the sampler varies are not the quantities you want to publish.
Nobody reports an MGE ellipticity component; they report an Einstein radius, a
magnification, a flux.

Those derived quantities are **latent variables**: computed from each sample of
the model rather than sampled directly, which means they arrive with full
posteriors rather than a single number propagated from a best fit. This
pipeline's ``util.AnalysisImaging`` declares the catalogue that gets computed —
the library latents enabled in ``config/latent.yaml`` (total lens, lensed-source
and source fluxes, both raw and in microJansky, plus the magnification and the
effective Einstein radius) together with four Euclid-specific aperture fluxes.

The aperture fluxes are the matched-aperture photometry described under "__The
FITS Header Contract__": the model lens image, convolved to the worst-seeing
band's resolution and summed within 1, 2, 3 and 4 times that band's PSF FWHM,
converted through ``MAGZERO`` into microJansky. They exist so that Euclid VIS
fluxes can be compared like-for-like against the other bands in an SED fit.

``scripts/tools/diagnose_latent.py`` replays this whole catalogue on a finished result
and prints every value, flagging NaNs and zero sentinels, without running a
search. It is the fastest way to check that a fit produced usable science.

__SLaM: Source, Light And Mass__

Everything above describes the *initial* lens model, which is two searches. The
full pipeline is five, and the idea that organises them is called **SLaM** —
Source, (lens) Light and Mass.

The reasoning is that one giant fit of the final, complex model is the worst
possible way to get to it. A model with a ``PowerLaw`` mass, a detailed lens
light and a pixelized source has too many parameters and too many degeneracies
for a sampler handed nothing but priors. It will find a local maximum, and it
will find a different one on the next lens.

So the model is built in stages, each seeded from the last:

- **Source first.** Complex mass models need a pixelized source, and a pixelized
  source needs a decent mass model to be initialised safely. So the chain starts
  by establishing a robust source using a *simple* mass model — an Isothermal
  with shear, exactly the ``vis_lp`` fit described above — and then upgrades that
  source to a pixelization.
- **Light second.** Modeling the lens light accurately means deblending it from
  the lensed source, which is only possible once the source is well described.
  With source and mass held fixed, the light model can be refined on its own.
- **Mass last.** The most complex mass model is fitted at the end, when the
  source and lens light are already known and the sampler's whole budget can go
  into the parameters that were the point of the exercise.

Each stage is a small, well-conditioned search whose result narrows the priors of
the next. That is *search chaining*, and it is why a five-search chain converges
where a one-search fit does not.

``scripts/full_model.py`` implements this chain — SOURCE LP, two pixelized SOURCE
PIX stages, LIGHT LP, and a MASS TOTAL stage fitting a ``PowerLaw`` plus shear —
with a prose block introducing each stage in turn. Read it for the design
reasoning in depth. For chaining as a general technique, outside this pipeline,
see ``scripts/guides/modeling/chaining.py`` and
``scripts/guides/modeling/slam_start_here.py`` in the autolens_workspace.
"""

"""
__Where To Go Next: The Fitting Pipelines__

``scripts/`` holds the pipelines. All of them are run from the repository root
and take the arguments listed above. ``scripts/README.md`` describes each in
full; one line each:

- ``scripts/initial_lens_model.py`` — **the entry point**, and what this file
  runs. MGE lens light + SIE + shear mass + MGE source (``vis_lp``), then a
  pixelized Delaunay source (``vis_pix``).
- ``scripts/sersic_lens_model.py`` — Sersic lens and source fits with the mass
  model fixed to the initial fit, giving the cleaner photometry that SED fitting
  needs. Chains off ``initial_lens_model`` run with ``--stage vis_lp``.
- ``scripts/lens_model_waveband.py`` — fits the lower-resolution NIR and EXT
  bands with the VIS lens model held fixed.
- ``scripts/sersic_lens_model_waveband.py`` — the **SED chain** driver, running
  the previous three in order over every band. Give it its own
  ``PYAUTO_OUTPUT_DIR`` so its many per-band results stay out of the main tree.
- ``scripts/mge_lens_only.py`` — MGE subtraction of the lens light only, which
  reveals the lensed source quickly for inspection without fitting a mass model.
- ``scripts/full_model.py`` — the full SLaM chain described above, ending in a
  ``PowerLaw`` + shear mass model.

``scripts/simulator.py`` is the only producer of simulated data here. No fitting
script auto-simulates a missing dataset, deliberately: most users fit real
Euclid imaging, and a silent auto-simulate would hide a broken data path.

__Where To Go Next: Workflow__

Once you have fitted more than a few dozen lenses, opening ``output/`` folder by
folder stops being a workflow. ``workflow/`` is the answer to that.

Its scripts use the PyAutoFit **database aggregator** to load every result under
``output/`` at once — querying by pipeline tag and search name, exactly the
``initial_lens_model`` / ``vis_lp`` strings described above — and distil them
into a few files:

- ``workflow/csv_make.py`` — ``.csv`` catalogues of lens model parameters across
  the whole sample, for scientific interpretation and for sharing.
- ``workflow/png_make.py`` — custom ``.png`` summaries, for example every fit
  laid out one lens per row for fast visual triage.
- ``workflow/fits_make.py`` — ``.fits`` images of the results, such as the
  deblended lens and lensed-source light.

The aggregator can also read from a ``.sqlite`` database rather than the
directory tree, which is worth doing once you are past a few hundred fits.
``workflow/example/`` holds the real versions used on Euclid runs.

__Where To Go Next: Catalogue__

``catalogue/`` is the stage after that: it turns a directory of finished fits
into the **inspection bundle** — one folder per lens holding the thirteen files a
scientist needs to judge that lens — plus the master CSVs spanning the sample.
It is the direct ancestor of the exported DR1 catalogue.

Its producers live in ``catalogue/scripts/``: they generate the mass, lens-Sersic,
source-Sersic and magnitude CSVs and the deblending FITS from the fits, and
collect the PNGs the searches already wrote. ``scripts/build_inspection_bundle.sh``
runs all seven stages in order for a sample. ``catalogue/README.md`` has the
file-to-producer table, the run order, and the upstream fit each stage needs.

__Further Reading__

- `The PyAutoLens readthedocs <https://pyautolens.readthedocs.io/en/latest>`_ —
  an overview of the core features, a new-user starting guide and an
  installation guide.
- `The autolens_workspace <https://github.com/PyAutoLabs/autolens_workspace>`_ —
  example scripts and the HowToLens lecture notebooks. The examples cited above
  live under ``scripts/imaging/features/`` and ``scripts/guides/modeling/``.
- `This repository's README <https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline>`_
  — installation, the argument table, the dataset requirements, the CI layout and
  the visualization settings.

__Questions__

If the output your science needs is not being produced, that is treated as a gap
in the pipeline rather than something for you to work around: contact **James
Nightingale on the Euclid Consortium Slack** so it can be added and become a
standard output of the Euclid strong lens modeling pipeline, and therefore of the
data release.

**PyAutoLens** also has automated pipelines for group-scale lenses, lensed point
sources such as quasars, and double source plane lenses. They were once in this
repository and were removed when it was narrowed to the Euclid VIS chain above;
they still live in **PyAutoLens** itself and can be restored here on request —
ask on Slack.
"""


if __name__ == "__main__":
    (
        sample_name,
        dataset_name,
        iterations_per_quick_update,
        number_of_cores,
        use_cpu,
        stage,
    ) = util.parse_fit_args()
    fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        number_of_cores=number_of_cores,
        use_cpu=use_cpu,
        stage=stage,
    )
