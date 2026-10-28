# Execution Configuration (`execution.json`)

The `execution.json` configuration object dictates the hardware utilization, concurrency limits, and memory management strategies for the `rasterCube` spatial processing pipeline. By abstracting these thresholds, the pipeline dynamically scales its I/O and CPU routines to match the host environment without hardcoding system limits into the core execution logic.

Place this file in the root of the `bmc` utility module (e.g., `src/bmc/execution.json`).

## Standard Configuration

```json
{
    "min_slices_for_parallel": 16,
    "max_safe_threadpool_workers": 8,
    "large_slice_threshold_mb": 50.0,
    "gdal_warp_block_size": 2048,
    "warp_cache_cleanup_delay_sec": 30
}
```

## Configuration components

### `min_slices_for_parallel`
*   **Default Value:** `16`
*   **Type:** Integer
*   **Mechanism:** Determines the threshold at which the pipeline transitions from sequential execution to parallelized execution. If a temporal stack or Z-axis group contains fewer slices than this value, the pipeline warps the data sequentially on the main thread.

### `max_safe_threadpool_workers`
*   **Default Value:** `8`
*   **Type:** Integer
*   **Mechanism:** Enforces a hard cap on the number of concurrent Python threads spawned when processing multiple small raster slices. 
*   **Tuning Rationale:** This acts as a critical safety valve. Python's Global Interpreter Lock (GIL) combined with concurrent file I/O operations scales poorly beyond 8-12 threads, leading to severe disk thrashing and memory spikes. Increase this value (e.g., 16-32) only if executing on a high-concurrency cloud worker with fast NVMe local storage and high RAM availability.

### `large_slice_threshold_mb`
*   **Default Value:** `50.0`
*   **Type:** Float
*   **Mechanism:** Defines the memory boundary (in Megabytes) for a single 2D raster array. Slices smaller than this threshold are distributed across multiple Python workers. Slices larger than this threshold bypass Python-level concurrency; instead, the pipeline processes slices sequentially but allocates all available CPU cores directly to GDAL's native C++ multithreading engine.
*   **Tuning Rationale:** Processing massive spatial grids concurrently via Python workers easily causes Out-Of-Memory (OOM) crashes as each worker provisions independent I/O buffers. If running on a memory-constrained container (e.g., Kubernetes pods with 8GB limits), lower this value to force native C++ threading sooner.

### `gdal_warp_block_size`
*   **Default Value:** `2048`
*   **Type:** Integer
*   **Mechanism:** Sets the internal block size (in pixels) for xarray and Dask chunking during the final cube materialization (e.g., `chunks={'x': 2048, 'y': 2048}`).
*   **Tuning Rationale:** A block size of 2048 balances RAM usage during lazy evaluation with efficient Zarr/NetCDF4 chunk writing. Increase to 4096 for extremely high-resolution continental grids to reduce metadata overhead, provided sufficient RAM is available for the larger chunk arrays.

### `warp_cache_cleanup_delay_sec`
*   **Default Value:** `30`
*   **Type:** Integer
*   **Mechanism:** Defines the sleep duration (in seconds) implemented immediately after the xarray materialization phase and garbage collection, before yielding the pipeline back to the orchestrator.
*   **Tuning Rationale:** High-throughput processing creates localized race conditions where asynchronous disk writing or network handlers fail to release file locks instantaneously. This buffer ensures all cached temporary files, NetCDF/Zarr lock files, and memory mapped pointers are fully flushed to disk and released by the OS before the next variable initiates. Reduce this to `5` or `10` on local SSDs; maintain at `30` for cloud or network-attached storage (NAS).