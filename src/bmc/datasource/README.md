# Datasource Module

This module serves as the acquisition layer for the cubing engine. It contains provider-specific clients and utility functions designed to abstract the complexities of different geospatial APIs.

Its primary responsibilities are:
1.  **Authentication**: Securely handling credentials for external services.
2.  **Taxonomic Harmonization**: Resolving biological names against standard backbones (e.g., GBIF, Catalogue of Life).
3.  **Query Generation**: Dynamically constructing syntactically correct API payloads, SQL strings, and spatial/temporal filters based on user configurations.
4.  **Orchestration**: Submitting data requests, polling for completion, and downloading raw assets to the staging area (`lake`).

## Supported Submodules

### CHELSA (`chelsa`)
Interfaces with the Cloud High-Resolution Environmental Modeling into the Future (CHELSA) data hosted on public S3-compatible storage.

*   **S3 Cataloging**:
    *   Utilizes high-speed S3 pagination to rapidly discover assets, bypassing slow recursive folder crawling.
    *   Extracts temporal (daily, monthly, annual), categorical, model (CMIP6), and scenario (SSP) metadata directly from object keys and filenames.
    *   Generates GDAL-compatible virtual file system (`/vsicurl/`) paths for optimized, lazy loading of Cloud Optimized GeoTIFFs (COGs).
*   **STAC & GeoParquet Orchestration**:
    *   Implements a multithreaded scraping engine via `ThreadPoolExecutor` to build standardized SpatioTemporal Asset Catalog (STAC) Items in memory asynchronously.
    *   Handles planetary spatial extents and complex temporal extractions across varying data levels (climatologies, bioclim, etc.).
    *   Leverages `stac-geoparquet` to compress memory-resident STAC objects into highly efficient, flattened binary GeoParquet databases for scalable catalog management.

### GBIF (`gbif`)
Provides high-speed interfacing with the Global Biodiversity Information Facility.

*   **Taxonomic Interface**:
    *   High-speed, parallelized backbone lookups using `ThreadPoolExecutor` and `lru_cache`.
    *   Name resolution against the GBIF Backbone or Catalogue of Life (CoL) hosted checklists.
    *   Automatic partitioning of accepted taxa from mismatches (fuzzy matches, higher-rank matches, or failures).
    *   Cross-mapping utilities between GBIF backbone keys and standard CoL identifiers.
*   **Query Helpers**:
    *   Generation of optimized GBIF SQL strings, including Selects, Select-Transforms (for CoL support), and complex WHERE clauses.
    *   Dynamic bounding box optimization to reduce query load via pre-filtering and `GBIF_Within` WKT constraints.
    *   Support for occurrence aggregation and gridding (EEA, EuroStat, DMSG, EQDG, etc.).
    *   Automatic mapping of specific taxon keys (integer or string) to optimal indexed GBIF SQL column positions.
*   **Orchestrator**: Wrapper functions to submit SQL queries, poll GBIF metadata for status changes, and fetch completed downloads to a target directory.

### WEkEO (`wekeo`)
Provides harmonized access to Copernicus and Sentinel data via the WEkEO HDA API.

*   **Query Generation**: Safely constructs dynamically structured JSON payloads for the HDA API. It automatically handles dataset-specific API quirks (e.g., proper routing for CORINE Land Cover vs. High Resolution Layers).
*   **Schema Retrieval**: Fetches metadata schemas and valid subsetting parameters (variables, constraints) directly from the WEkEO HDA broker.
*   **Inventory Crawling**: An advanced utility that crawls the WEkEO API to test every valid combination defined in a dataset's schema (productType, year, format, resolution). It builds a distinct inventory record for every available asset, including a spatial check to verify if the dataset supports bounding box subsetting.