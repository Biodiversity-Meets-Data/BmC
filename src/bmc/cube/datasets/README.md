# Datasets Submodule

The `datasets` submodule is the designated storage location for the concrete implementations of the `rasterCube` and `vectorCube` child classes. It is logically partitioned into `raster` and `vector` directories to separate continuous environmental processing engines from discrete, event-based spatial mapping engines.

## Raster Implementations

The `raster` directory contains implementations that inherit from the `rasterCube` base class. 

### CHELSA (`chelsaCube`)
The `chelsaCube` class handles the ingestion and processing of Cloud High-Resolution Environmental Modeling into the Future (CHELSA) data. 
* It is dedicated to the single-pass generation of high-resolution CHELSA V2.1 spatiotemporal data cubes.
* It filters the CHELSA GeoParquet inventory to generate a precise execution queue based on YAML configurations.
* It maps physical environmental variables to strict GDAL resampling algorithms to prevent data corruption. For example, it applies `bilinear` resampling for continuous gradients like air temperature ('tas') and `average` resampling for volume conservation like precipitation ('pr').
* It utilizes a "Metadata Catch and Release" architecture to translate flat tabular metadata into rich, multi-dimensional Xarray coordinates (such as CMIP6 ensembles or time series).
* It enforces minimum atmospheric resolutions when resolving target grids, clamping metric grids to a minimum of 1km and WGS84 degree grids to 15sec.

## Vector Implementations

The `vector` directory contains implementations that inherit from the `vectorCube` base class.

### GBIF (`gbifCube`)
The `gbifCube` class orchestrates the generation of Global Biodiversity Information Facility (GBIF) data cubes.
* It is dedicated to generating and processing GBIF spatiotemporal vector data cubes by translating abstract YAML recipes into executable GBIF Hive SQL queries.
* It handles the ingestion of raw occurrence data and can dynamically generate spatial topologies (Points, Polygons, or Point Clouds) based on coordinate uncertainty.
* It supports an `api_cube` processing mode that offloads spatial aggregation directly to the GBIF SQL servers. 
* It parses implicit and explicit GBIF grid cell codes (such as EEA and EQDG formats) into exact geometrical boundaries represented as Shapely Polygons.
* It natively mines Coordinate Reference Systems (CRS) from grid cell codes, mapping explicit strings or implicit visual patterns to standard EPSG identifiers.