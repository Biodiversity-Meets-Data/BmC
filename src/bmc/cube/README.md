# Cube Module Overview

The `cube` module is the core generation engine of the `bmc` spatiotemporal pipeline. It is responsible for orchestrating, standardizing, and generating mathematically aligned data cubes from disparate raw ecological data. 

To enforce the separation of concerns and maintain a scalable architecture, the module is partitioned into three primary submodules: **Spatiotemporal**, and **Datasets**, **Orchestrator**.

---

## 1. Spatiotemporal Submodule
**Goal:** Provide the architectural blueprints and Abstract Base Classes (ABCs) that define how spatial data is handled, warped, and standardized.

This submodule ensures that all datasets behave consistently regardless of their origin. It enforces consistent dual-stream logging, hardware telemetry, topological sanitization, and mathematical resolution against a master `GRID_REGISTRY`.

**Key Functionality:**
*   **`cube.py` (Parent Base):** Centralizes non-spatial boilerplate functionality, including logging, directory topology provisioning, and resource profiling.
*   **`raster.py` (Raster Base):** The engine for **continuous** environmental variables. It manages out-of-core spatial warping via GDAL C++ multithreading, grid snapping, and metadata injection to reassemble 2D TIFFs into 3D/4D arrays.
*   **`vector.py` (Vector Base):** The engine for **discrete, event-based** spatial mapping. It manages geometry sanitization, DuckDB spatial pushdown, mass-conserving translation into target templates, and outputs a highly structured relational star schema (Dimensions and Fact tables).

---

## 2. Datasets Submodule
**Goal:** House the concrete, vendor-specific implementations of the `rasterCube` and `vectorCube` base classes.

This submodule is logically partitioned into `raster` and `vector` directories. Each script here acts as an adapter, translating the strict rules of the `spatiotemporal` base classes into the specific API logic required by external data providers.

**Key Functionality:**
*   **Raster Implementations (e.g., `chelsaCube`):** Handles the ingestion of CHELSA V2.1 data. It maps physical environmental variables to strict GDAL resampling algorithms (e.g., `bilinear` for temperature, `average` for precipitation) and parses CMIP6 ensembles into Xarray coordinates using a metadata catch-and-release architecture.
*   **Vector Implementations (e.g., `gbifCube`):** Handles the ingestion of GBIF occurrence data. It translates YAML recipes into executable GBIF Hive SQL queries, dynamically generates spatial topologies (Points, Polygons, Point Clouds), parses native grid cell codes (EEA, EQDG), and supports `api_cube` modes that offload spatial aggregation to the provider's servers.

---

## 3. Orchestrator Submodule
**Goal:** Serve as the central command center for the end-to-end execution of ecological data pipelines. 

The `bmd_cube` class resides here and is responsible for fully automating the generation of unified datasets. It acts as the bridge between user-defined YAML configuration recipes and the underlying processing engines. 

**Key Functionality:**
*   **Dynamic Dispatch & Async Execution:** Automatically routes processing tasks to the correct raster or vector engines and optimizes pipeline speed by submitting heavy remote API requests in the background while processing local tasks.
*   **Output Tree Generation:** Generates a highly organized directory structure that clearly separates harmonized data cubes (exported as Zarr, NetCDF, or GeoParquet) from metadata.
*   **Reproducibility & Provenance:** Dumps a frozen copy of the executed recipe and generates a strict `provenance_metadata.json` audit log capturing system environments.
*   **STAC Cataloging:** Wraps disparate dataset outputs into cohesive, hierarchically structured PySTAC Catalogs (`catalog.json`).