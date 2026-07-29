# Data Cube Engine: Raster & Vector Base Classes

This repository contains the foundational architectural blueprints for constructing massive, spatiotemporal ecological data lakes and multidimensional cubes. The engine is bifurcated into two primary Abstract Base Classes (ABCs): **`raster_cube`** and **`vector_cube`**. 

These base classes handle the complex mathematical alignment, distributed memory management (Dask/DuckDB), spatial topological transformations, and strict Standards-compliant metadata generation (STAC, CF Conventions) required for global-scale environmental modeling.

---

## 1. Raster Engine (`raster_cube`)

The `raster_cube` class defines the pipeline for continuous spatial data (e.g., satellite imagery, gridded climate projections). It manages out-of-core GDAL warping, dynamically scaling hybrid multi-threading, and N-dimensional coordinate stacking (via `xarray`).

### Core Principles
* **Out-of-Core Processing:** Leverages Dask and optimized GDAL C++ multithreading to warp slices that exceed physical RAM limits.
* **Strict Grid Alignment:** Forces disparate native datasets into a rigidly defined master coordinate grid (CRS, resolution, and physical bounds) using affine reprojection.
* **Metadata Injection:** Automatically expands 2D TIFFs into 3D/4D arrays (e.g., time, ensemble, depth) based on harvested STAC catalog metadata.
* **CF Standards & STAC:** Exports self-describing Zarr/NetCDF structures mapped via PySTAC.

### Requirements for Implementing a Raster Child Class
To integrate a new raster data source (e.g., WEkEO, CHELSA, Planetary Computer), the child class must inherit from `raster_cube` and implement the following five `@abstractmethod` routines:

#### 1. `resolve_target_grid(spatial_cfg, logger) -> str`
Translates arbitrary user-defined spatial strings from the YAML recipe (e.g., `resolution: 1km`) into a strictly validated dictionary key corresponding to the parent engine's `GRID_REGISTRY`. This ensures flawless mathematical projection mapping.

#### 2. `generate_execution_plan(recipe, logger) -> pd.DataFrame`
Bridges the gap between the YAML recipe and the remote STAC catalog. Must return a Pandas DataFrame acting as an execution queue with mandatory columns:
* `level`: The processing family (e.g., 'climatologies').
* `variable`: The physical variable name (e.g., 'pr', 'tas').
* `vsi_path`: The Virtual File System (VSI) path or URL to the target asset.

#### 3. `parse_metadata(row, da) -> Tuple[str, xr.DataArray]`
Extracts variable-specific dimensional metadata (e.g., extracting "Year=2020", "Month=July" from the file name or row metadata) and injects it into the 2D DataArray as new Z-axis coordinates. Essential for reassembling the 3D stack.

#### 4. `get_resample_rule(variable_name) -> str`
Defines the ecological/mathematical integrity rule for interpolation. Must map a variable string to a GDAL C++ integer constant (e.g., returning `'nearest'` for categorical Land Cover, or `'bilinear'` for continuous Temperature).

#### 5. `apply_multi_index(level, dataset) -> xr.Dataset`
Compiles independent dimensional coordinates (like 'year' and 'month') into a unified Pandas MultiIndex structure on the Z-axis, finalizing the multidimensional shape of the NetCDF/Zarr.

---

## 2. Vector Engine (`vector_cube`)

The `vector_cube` class governs discrete, event-based spatial mapping (e.g., species occurrences, protected area polygons). It handles the mathematical topology required to translate relational vectors into continuous grid matrices.

### Core Principles
* **Spatial Pushdown:** Utilizes DuckDB spatial extensions and Dask-GeoPandas to execute bounding box intersections on remote Parquet files *before* loading them into memory.
* **Topological Sanitization:** Automatically heals invalid geometries, drops empty shapes, and forces planar 2D limits.
* **Mass Conservation:** Translates varying vector topologies (Points, Polygons, Point Clouds) to target templates while preserving metrics using fractional mapping and weighted areal allocations.
* **Relational Output:** Outputs high-performance Star-Schema GeoParquets (dimension tables linked to fact tables) registered as STAC Items.

### Requirements for Implementing a Vector Child Class
To integrate a new discrete vector provider (e.g., GBIF, IUCN, eBird), the child class must inherit from `vector_cube` and implement the following two `@abstractmethod` routines:

#### 1. `resolve_target_grid(spatial_cfg, logger) -> str`
Identical in purpose to the raster equivalent. Maps arbitrary spatial configurations from the user's recipe to a master grid key registered in `GRID_REGISTRY` to lock in projection boundaries and resolutions.

#### 2. `fetch_data(recipe, logger=None) -> gpd.GeoDataFrame`
The core ingestion function. Must query the vendor's specific database, STAC API, or file directory based on constraints in the `recipe` (e.g., temporal windows, taxa IDs) and return a raw, uniform `geopandas.GeoDataFrame`. The base class pipeline will handle everything downstream (sanitization, clipping, gridding, and aggregation).

---

## Summary of Execution Flow
Both pipelines rely heavily on a `process_cube(recipe, logger)` orchestration method provided by the base class. As a developer, **you do not need to rewrite the extraction, clipping, warping, or export logic**. By successfully implementing the abstract methods above, your child class seamlessly plugs into the highly optimized, memory-safe data cube pipeline.
