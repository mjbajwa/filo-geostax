# filo-geostax

Some geostatistics code using naive Gaussian Processes in tinygp and JAX, on Filo's open drillhole dataset. 

## Background

Filo has released roughly ~60,000 drill core assay data, at roughly 1 meter vertical resolution for each drill hole (around 250 drill holes). We are interested in inferring the copper composition of the whole deposit.

We are particularly interested in using naive Gaussian Processes and seeing how far this approach can take us on modern GPUs. We are able to use ~30,000 data points as our training data without running out of memory on an RTX A4500. Some additional JAX code is written to predict in a batch manner over 1 million cells, by moving between CPU and GPU (see `src/utils/model.py`)

# Sample output

A draw from the fitted GP over the full region is plotted using Paraview:

![Sample 3D fitted GP draw and drill holes](./docs/static/240412_results.png)

## TODO: 

1. Improve topography estimate of the earth using open source datasets. Right now I infer topography using a GP model based on the elevation data in the "collars" survey of the drill holes.

2. Improve grid generation sub-methods. Right now the X-Y discretization is done in a place that it shouldn't be (topography codebase). Refactor into a useful module (where can I use some of `pygslib` computations?)

3. Co-kriging for multiple elements (Cu and Ag in Filo's case)

4. Active learning (Bayesian Optimization) for drill-hole optimization?

5. Comparison with variogram estimating methods