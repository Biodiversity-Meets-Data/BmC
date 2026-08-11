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