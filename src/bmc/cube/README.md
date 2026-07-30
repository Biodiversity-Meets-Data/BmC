# BMD Cube Orchestrator

The `bmd_cube` class serves as the central orchestrator for constructing multi-dimensional spatiotemporal data cubes. It is designed to manage the end-to-end execution of ecological data pipelines, dynamically dispatching tasks to specific raster and vector engines.

---

## High-Level Overview

By parsing YAML configuration recipes, `bmd_cube` completely automates the generation of unified datasets. It handles asynchronous data fetching, localized dataset warping, and metadata generation, culminating in a highly structured and reproducible output directory. 

**Key Capabilities:**
*   **Dynamic Dispatch:** Automatically routes processing tasks to designated engines like CHELSA (raster) and GBIF (vector) based on YAML configurations.
*   **Asynchronous Execution:** Optimizes pipeline speed by submitting heavy remote API requests (like GBIF) to run in the background while simultaneously downloading and processing synchronous local tasks.
*   **Strict Reproducibility:** Automatically dumps a frozen copy of the executed `recipe.yaml` and generates a rigid `provenance_metadata.json` audit log capturing system environments and execution timestamps.
*   **STAC Integration:** Wraps disparate dataset outputs into cohesive, hierarchically structured PySTAC Catalogs (`catalog.json`) to ensure machine-readable spatial metadata.

---

## Output Directory Structure

![Datatree](../../../img/datatree.png)