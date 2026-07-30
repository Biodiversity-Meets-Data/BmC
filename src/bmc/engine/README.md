# BMC Spatial Engine Overview

This repository contains the core components of the spatiotemporal pipeline for processing and aligning continuous and discrete ecological datasets. The architecture is built around a foundational grid registry and specialized engines for both raster and vector data. 

Below is a general overview and introduction to the three primary classes that govern the spatial physics, geometric transformations, and master grid alignments within the system.

---

## 1. `base_spatial_grid`
**Source File:** `bmc/engine/base.py`

The `base_spatial_grid` class acts as the foundational spatial truth of the entire spatiotemporal pipeline. It provides the mathematical blueprints inherited by both the raster and vector engines to guarantee perfect 1-to-1 pixel alignment across datasets.

### Key Features & Responsibilities:
* **The Master Grid Registry (`GRID_REGISTRY`):** Holds the definitive registry of all supported master grids. This includes predefined specifications for the EEA Reference Grid (EPSG:3035), Global Equal Area (EPSG:6933), and Global WGS84 (EPSG:4326), explicitly defining their Coordinate Reference Systems (CRS), native pixel resolutions, and absolute bounding boxes.
* **Dynamic Grid Resolution:** Dynamically constructs and validates master grid keys from user configurations (e.g., combining "EEA" and "10km" into "EEA_10km").
* **Safe Fetch Envelopes:** Constructs densified, buffered source envelopes guaranteed to fully encapsulate target grid regions, preventing edge starvation or NaN boundary artifacts during reprojection.
* **Aligned Raster Templates:** Generates empty, mathematically rigid `xarray.DataArray` templates that are perfectly snapped to a predefined master grid's intervals.
* **Deterministic Global Indexing:** Converts local 2D spatial coordinates into a deterministic, globally consistent 1D index based on a master grid's absolute origin.

---

## 2. `raster_engine`
**Source File:** `bmc/engine/raster_engine.py`

Inheriting from `base_spatial_grid`, the `raster_engine` class is the fundamental spatial physics and geometric transformation engine for raster arrays. It is fully decoupled from data lifecycle management, focusing exclusively on the physical transformations of arrays.

### Key Features & Responsibilities:
* **GDAL Integration:** Provides a universal spatial toolbox utilizing out-of-core GDAL Python wrappings and coordinate registries. It maintains a registry mapping human-readable resampling algorithms (e.g., "bilinear", "nearestNeighbour", "average") to their corresponding GDAL C++ integer constants.
* **Spatial Geometry Sanitization:** Validates and sanitizes `xarray` spatial metadata, standardizing horizontal/vertical axes and enforcing CRS to ensure GDAL and `rioxarray` compatibility.
* **Out-of-Core Affine Reprojection:** Performs spatial reprojections and snaps raster data to strictly defined master grids. It intelligently handles both in-memory `xarray` objects (by streaming them to disk) and physical raster files.
* **Fractional Continuous Area Calculation:** Contains pure spatial math primitives to isolate single categorical classes and compute their fractional coverage. It dynamically routes large disk-based Virtual Rasters (VRTs) to GDAL streaming and small datasets to native `xarray` math.

---

## 3. `vector_engine`
**Source File:** `bmc/engine/vector_engine.py`

Also inheriting from `base_spatial_grid`, the `vector_engine` class handles all vector-specific geometrical operations. It maps vector datasets directly onto the mathematically rigid master grid templates defined by the base class.

### Key Features & Responsibilities:
* **Coordinate to Geometry Conversion:** Converts tabular spatial records into GeoDataFrames. It handles uncertainty by either applying geometric buffers or generating Monte Carlo point clouds based on coordinate uncertainty measurements.
* **Geometry Sanitization & Topology Healing:** Cleans, flattens, normalizes, and validates dirty vector geometries. It forces 2D planar geometries and uses GEOS algorithms to fix broken topologies.
* **Spatial Point Clouds:** Generates memory-efficient spatial point clouds around feature centroids using specified distributions (e.g., Gaussian or Uniform) to account for coordinate uncertainty.
* **Template Mapping:** Provides specialized methods to map various vector formats to the master grid template:
    * **Points:** Executes point geometry intersections using spatial joins or KDTree nearest-neighbor queries.
    * **Point Clouds:** Computes probabilistic point cloud distributions (Monte Carlo procedure) to map fractional weights onto the spatial topology blueprint (point distribution over raster) or to assign points to grid cells based on majority ruling.
    * **Polygons:** Maps polygon datasets by perfectly fracturing and distributing intersection weights (areal fractions) onto the grid blueprint.
    * **Cell Collections:** Maps pre-aggregated, perfectly aligned grid cell polygons directly to the template.
* **Vector Transformation QA/QC:** Validates the mathematical and topological integrity of spatial transformations, specifically tracking mass conservation (fractional mapping) and topological drift (classification).
