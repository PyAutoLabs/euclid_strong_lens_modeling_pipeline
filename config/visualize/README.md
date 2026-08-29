The `config` folder contains configuration files which customize default **PyAutoLens**.

# Files

- `general.yaml`: Customizes general visualization settings (e.g. the matplotlib backend).
- `include.yaml`: Customize features that appears on plotted images by default (e.g. a mask, a grid).
- `plots.yaml`: Customize which figures are output during a model-fit.
- `plots_search.yaml`: Customize which non-linear search figures are output during a model-fit.
- `mat_wrap.yaml`: Specify the default matplotlib settings when figures and subplots are plotted.
- `mat_wrap_1d.yaml`: Specify the default matplotlib settings when 1D figures and subplots are plotted.
- `mat_wrap_2d.yaml`: Specify the default matplotlib settings when 2D figures and subplots are plotted.

# Changing the colormap

Every 2D figure this pipeline produces — imaging data, fits, residual maps,
inversion reconstructions — draws with the colormap named by the `colormap` key
of `general.yaml`. This repository ships **magma**:

```yaml
colormap: magma   # any matplotlib colormap name, or `autoarray` for the bundled PyAuto colormap
```

Editing that one key changes every 2D figure the pipeline produces. `autoarray`
is the colormap bundled with **PyAutoArray**; any other value is looked up in
matplotlib, so `magma`, `viridis`, `inferno`, `plasma`, `jet` and the rest of
`list(matplotlib.colormaps)` all work.

A name matplotlib does not recognise (a typo, say) raises a `ValueError` naming
the key and the offending value — it is **not** silently swapped back for the
default, so a colormap setting never goes quietly ignored.

## One figure at a time

To override the colormap for a single figure without touching config, pass
`colormap=` to the plot function:

```python
import autolens.plot as aplt

aplt.plot_array(array=image, colormap="viridis")
aplt.subplot_fit_imaging(fit=fit, colormap="inferno")
aplt.subplot_tracer(tracer=tracer, grid=grid, colormap="inferno")
```

A few figures deliberately fix their colormap because the colormap carries
meaning a global preference should not override — the `array_overlay` of
`plot_array` (`Greys`, so it stays legible over the main array), and the
weak-lensing position-angle (cyclic) and residual (diverging) maps. The full
list is in **PyAutoArray**'s `autoarray/config/visualize/README.md`.
