The `image_mesh` folder contains configuration files for the default priors assumed for `image_mesh` objects.

These model components construct the (y,x) grid of coordinates used for a mesh in the image-plane.

For example, the `Hilbert` image-mesh computes the centres of the image mesh by drawing points from the Hilbert
curve of an adapt image, concentrating them in the image's brighter regions. The `Overlay` image-mesh instead lays
a uniform grid of the requested shape over the masked region.
