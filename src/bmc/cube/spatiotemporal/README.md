# Spatiotemporal Data Cube Submodule

This directory serves as the core architectural heart of the `bmc` spatiotemporal engineering pipeline. It provides the **Abstract Base Classes (ABCs)** used to define and execute consistent workflows for translating disparate raw ecological data (both raster and vector) into highly optimized, mathematically aligned data lakes and data cubes.

This module enforces consistent logging, hardware telemetry, spatial framework resolution against the `GRID_REGISTRY`, topological sanitization, and Standards-compliant metadata generation (STAC, CF Conventions) across all inherited dataset orchestrators.

## Submodule Structure

The module is organized around a parent blueprint that branches into specialized raster and vector engines:

-   **`cube.py` (Parent Base: `dataCube`)**: The abstract parent of both Raster and Vector data cubes. It centralizes all non-spatial boilerplate functionality required for a data pipeline execution, including dual-stream logging (file + console), initial directory topology provisioning (nested logs/dataset folders), YAML recipe context parsing, and resource profiling via the `ResourceProfiler`.
-   **`raster.py` (Raster Base: `rasterCube`)**: Inherits from `rasterEngine`, `dataCube`, and `ABC`. This is the fundamental engine used for building **continuous** environmental variables from raw gridded data (e.g., satellite imagery, climate projections). It manages ingestion, out-of-core spatial warping and snapping to rigid master grids, dimensional metadata extraction for reassembling 2D TIFFs into 3D/4D arrays (NetCDF/Zarr), dynamic worker pool scaling, and raster-specific STAC Item generation.
-   **`vector.py` (Vector Base: `vectorCube`)**: Inherits from `vectorEngine`, `dataCube`, and `ABC`. This engine governs **discrete, event-based** spatial mapping (e.g., species occurrences, protected area polygons). It handles retrieving vector geometries (with DuckDB spatial pushdown for remote Parquet files), geometry sanitization, mass-conserving translation from varying topologies (Point, Polygon, Point Cloud) into target templates, aggregation logic, Star Schema GeoParquet output (Fact/Dimension), and vector STAC Item generation.

---

## Developer Guide: How to Extend these Bases

The cube base classes provide a template to integrating new data cubes for new data sources into the cubing engine functionality. For a more in depth discussion as to how the implement a new instance of both we refer to the tutorial section in this repository, particulary tutorial 2a and 2b.

# Catalog Schema Requirements (`generate_execution_plan`)

The pipeline strictly requires catalogs to be formatted as flat, columnar GeoParquet files. This ensures high-performance vectorized spatial and temporal querying.

## 1. Mandatory Core Columns
If any of these columns are missing, the execution plan generator will throw an error and terminate the pipeline.

| Column Name | Data Type | Engine Function |
| :--- | :--- | :--- |
| **`geometry`** | `geometry` (Shapely) | Evaluated using `.intersects()` to filter data matching the recipe's spatial bounding box. |
| **`datetime`** | `datetime64[ns, UTC]`| Timezone-aware timestamp queried against the recipe's start and end dates. |
| **`level`** | `str` | Categorical parameter grouping the temporal frequency or sub-collection (e.g., "daily", "monthly"). |
| **`variable`** | `str` | Categorical parameter identifying the environmental variable (e.g., "tas", "pr"). |
| **`vsi_path`** | `str` | The direct URI string to the physical file (e.g., `s3://...` or `/vsicurl/...`). Extracted and flattened from STAC assets. |

## 2. Standard Metadata Columns
Highly recommended. If present, the engine automatically extracts these for robust VRT generation and STAC Item rebuilding.

| Column Name | Data Type | Engine Function |
| :--- | :--- | :--- |
| **`proj:epsg`** | `int` | Validates that source tiles share the same projection before executing GDAL merges. |
| **`nodata`** | `float` | Used to enforce consistent missing-value representations during out-of-core warping. |

## 3. Dynamic Auxiliary Columns (Dataset-Specific)
Child classes can define custom dimensions via the `get_auxiliary_dimensions()` method. Any key returned by this method must perfectly match a flat column in the GeoParquet file.