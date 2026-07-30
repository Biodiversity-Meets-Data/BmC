# Biodiversity meets Cubes: the BMD Cubing Engine

A repository aimed at the development of the cubing engine that combines biodiversity data with abiotic data to produce data cubes ready for use within the VRE's of the [Biodiversity Meets Data (BMD)](https://bmd-project.eu/) project.

## Repository Setup

This project uses a dual-file packaging strategy to safely handle complex geospatial dependencies (like GDAL and Xarray) across different operating systems. 

* **`environment.yml`**: Uses Conda to install the heavy, C++ backed spatial libraries.
* **`pyproject.toml`**: Uses Pip to install the local cubing engine code as an importable Python package.

**Prerequisites:** You must have [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) installed on your system.

### Step 1: Build the geospatial environment

Navigate to the root of the cloned repository and create the Conda environment using the `.yml` file:

```bash
conda env create -f environment.yml
```

### Step 2: Activate the environment

```bash
conda activate BmC
```

### Step 3: Install the local cubing engine

```bash
pip install -e .
```
