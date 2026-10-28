import xml.etree.ElementTree as ET

def parse_aux_xml_stats(xml_content):
    """
    Parses GDAL .aux.xml files to extract band statistics.
    
    Extracts standard GDAL statistical metadata (minimum, maximum, 
    mean, standard deviation) commonly found in Copernicus auxiliary files.

    Parameters
    ----------
    xml_content : bytes or str
        The raw XML content of the .aux.xml file.

    Returns
    -------
    dict
        A dictionary mapping the cleaned statistic names to their float values.
        Returns an empty dictionary if parsing fails or no stats exist.
    """
    stats = {}
    try:
        root = ET.fromstring(xml_content)
        
        # GDAL stores statistics in <MDI key="STATISTICS_MEAN"> format
        for mdi in root.findall('.//MDI'):
            key = mdi.get('key')
            
            if key and key.startswith('STATISTICS_'):
                # Clean the key (e.g., 'STATISTICS_MEAN' -> 'mean')
                clean_key = key.replace('STATISTICS_', '').lower()
                try:
                    stats[clean_key] = float(mdi.text)
                except (ValueError, TypeError):
                    stats[clean_key] = mdi.text
                    
        return stats
    except Exception:
        # Return empty stats silently if the XML is malformed
        return {}

def parse_legend_colors(xml_content):
    """
    Parses QGIS Style (QML) or Styled Layer Descriptor (SLD) XML content to 
    extract official legend color mappings.

    This function processes XML styling metadata commonly associated with 
    geospatial raster datasets. It dynamically strips XML namespaces and 
    attempts to parse the document first as a QML file (searching for 
    `<paletteEntry>` nodes). If no QML entries are found, it falls back to 
    parsing it as an SLD file (searching for `<ColorMapEntry>` nodes). 
    
    The extracted mapping links discrete pixel values to their human-readable 
    labels and hex color codes.

    Parameters
    ----------
    xml_content : str or bytes
        The raw XML string or bytes content of the legend file to be parsed.

    Returns
    -------
    dict
        A dictionary mapping the integer pixel values to their corresponding 
        label and color. The structure is as follows:
        
        {
            1000: {"label": "Class Name", "color": "#HEXCODE"},
            ...
        }
        
        Returns an empty dictionary if the XML parsing fails, if the document 
        is malformed, or if neither `<paletteEntry>` nor `<ColorMapEntry>` 
        tags are found.

    Notes
    -----
    - XML namespaces (e.g., `{http://www.opengis.net/sld}`) are stripped 
      dynamically during traversal to ensure reliable tag matching.
    - Floating-point representations of class values (e.g., `quantity="1100.0"` 
      in SLD files) are safely parsed and cast to integers.
    - All exceptions during parsing are caught and suppressed to guarantee 
      pipeline stability, falling back to returning an empty dictionary.

    Examples
    --------
    Parsing a standard QML snippet:
    
    >>> qml_data = '''
    ... <qgis>
    ...   <paletteEntry value="1" label="Water" color="#0000FF"/>
    ...   <paletteEntry value="2" label="Forest" color="#00FF00"/>
    ... </qgis>'''
    >>> parse_legend_colors(qml_data)
    {1: {'label': 'Water', 'color': '#0000FF'}, 2: {'label': 'Forest', 'color': '#00FF00'}}

    Parsing an SLD snippet:
    
    >>> sld_data = '''
    ... <StyledLayerDescriptor>
    ...   <ColorMapEntry quantity="10.0" label="Urban" color="#AAAAAA"/>
    ... </StyledLayerDescriptor>'''
    >>> parse_legend_colors(sld_data)
    {10: {'label': 'Urban', 'color': '#AAAAAA'}}
    """
    try:
        root = ET.fromstring(xml_content)
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        legend = {}
        # Try QML
        for entry in root.findall('.//paletteEntry'):
            if entry.get('value'):
                legend[int(float(entry.get('value')))] = {"label": entry.get('label'), "color": entry.get('color')}
        # Try SLD
        if not legend:
            for entry in root.findall('.//ColorMapEntry'):
                if entry.get('quantity'):
                    legend[int(float(entry.get('quantity')))] = {"label": entry.get('label'), "color": entry.get('color')}
        return legend
    except Exception:
        return {}

import os

def discover_s3_assets_bfs(base_prefix, s3_client, bucket_name="eodata"):
    """
    Explores an S3 directory structure using a Breadth-First Search (BFS) 
    algorithm to discover and categorize geospatial assets.

    This function dynamically crawls an S3 bucket level-by-level using the 
    OpenStack Swift compatible `Delimiter='/'` parameter. This approach prevents 
    the silent server timeouts and empty responses that frequently occur when 
    performing deep, flat queries on massive or highly nested datasets 
    (like the Copernicus Data Space Ecosystem).

    As it navigates, it identifies and categorizes specific file types essential 
    for STAC catalog generation: main rasters (`.tif`), standard ISO metadata 
    (`.xml`), GDAL auxiliary statistics (`.tif.aux.xml`), and symbology legends 
    (`.qml` or `.sld`). Assets are grouped into a dictionary keyed by their 
    parent folder path.

    Parameters
    ----------
    base_prefix : str
        The root S3 prefix (folder path) from which to begin the BFS crawl. 
        It is highly recommended that this string ends with a forward slash 
        (`/`) to correctly align with S3 delimiter logic.
    s3_client : boto3.client
        The authenticated boto3 S3 client used to execute bucket queries.
    bucket_name : str, optional
        The target S3 bucket containing the assets. Defaults to "eodata".

    Returns
    -------
    tiles : dict
        A nested dictionary mapping the S3 folder path (the "tile" directory) 
        to its discovered assets. Structure:
        
        {
            "Path/To/Tile_A": {
                "raster": "Path/To/Tile_A/image.tif",
                "metadata_xml": "Path/To/Tile_A/metadata.xml",
                "aux_xml": "Path/To/Tile_A/image.tif.aux.xml"
            },
            ...
        }
        
    legend_data : str or None
        The exact S3 key (filepath) of the first discovered legend file 
        (ending in `.qml` or `.sld`). Returns `None` if no styling file 
        is found in the scanned hierarchy.

    Notes
    -----
    - **Grouping Logic:** The function assumes that all files belonging to a 
      single distinct geographical tile or observation reside within the same 
      terminal folder. 
    - **Pagination:** Uses `boto3.client.get_paginator` to safely handle S3 
      directories containing more than 1,000 files/folders at a single level.

    Examples
    --------
    Executing a crawl on a nested crop-type dataset prefix:

    >>> import boto3
    >>> s3 = boto3.client('s3', endpoint_url="https://eodata.dataspace.copernicus.eu")
    >>> base = "CLMS/landcover_landuse/crop_types/clms_vlcc_crop-types_europe_10m_yearly_v1/2017/"
    >>> tiles, legend = discover_s3_assets_bfs(base, s3, "eodata")
    >>> print(legend)
    'CLMS/landcover_landuse/crop_types/.../legend/clms_crop_legend.qml'
    
    >>> print(tiles['CLMS/landcover_landuse/.../2017/01/01/TILE_E09N27'])
    {
        'raster': '.../TILE_E09N27/CLMS_HRLVLCC_CTY_..._R00.tif', 
        'metadata_xml': '.../TILE_E09N27/CLMS_HRLVLCC_CTY_..._R00.xml', 
        'aux_xml': '.../TILE_E09N27/CLMS_HRLVLCC_CTY_..._R00.tif.aux.xml'
    }
    """
    folders_to_scan = [base_prefix]
    tiles = {}
    legend_data = None

    print("Crawling S3 directory structure...")
    while folders_to_scan:
        current_prefix = folders_to_scan.pop(0)
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=current_prefix, Delimiter='/'):
            # Enqueue subfolders for scanning
            for cp in page.get('CommonPrefixes', []):
                folders_to_scan.append(cp['Prefix'])
            
            # Categorize files
            for obj in page.get('Contents', []):
                key = obj['Key']
                filename = os.path.basename(key)
                folder = os.path.dirname(key)

                if filename.endswith('.qml') or filename.endswith('.sld'):
                    legend_data = key
                    continue

                if folder not in tiles: tiles[folder] = {}
                
                if filename.endswith('.tif') and not filename.endswith('.aux.xml'):
                    tiles[folder]['raster'] = key
                elif filename.endswith('.tif.aux.xml'):
                    tiles[folder]['aux_xml'] = key
                elif filename.endswith('.xml') and not filename.endswith('.aux.xml'):
                    tiles[folder]['metadata_xml'] = key

    return tiles, legend_data

def parse_iso_metadata(xml_content, filename=""):
    """
    Parses ISO 19115 XML metadata to extract spatial bounding boxes, 
    coordinate reference systems, spatial resolutions, and dataset lineage.

    This function processes the standardized XML metadata files provided 
    with Copernicus geospatial datasets. It strips namespaces dynamically 
    to simplify parsing and includes robust fallback mechanisms to handle 
    common metadata errors (such as tile IDs erroneously placed in the 
    spatial resolution tags).

    Parameters
    ----------
    xml_content : str or bytes
        The raw XML string or bytes content of the ISO 19115 metadata file.
    filename : str, optional
        The name of the dataset file or tile directory. Used as a secondary 
        fallback to extract the spatial resolution using regular expressions 
        if the XML tag is invalid or missing. Default is an empty string.

    Returns
    -------
    bbox : list of float or None
        The spatial bounding box in the format [west, south, east, north]. 
        Returns None if parsing entirely fails.
    epsg : int
        The Coordinate Reference System EPSG code (e.g., 3035). Defaults 
        to 4326 if not found.
    res : int or float
        The spatial resolution in meters. Defaults to 10 if extraction fails.
    abstract : str
        The descriptive abstract of the dataset. Defaults to "Copernicus Dataset" 
        if not found.
    lineage : str
        The processing statement or lineage of the dataset. Defaults to an 
        empty string if not found.
    """
    try:
        # Load the raw XML string into an ElementTree object for traversal
        root = ET.fromstring(xml_content)
        
        # Iterate through every element in the XML tree to strip namespaces.
        # ISO XMLs use heavy namespace prefixing (e.g., {http://www.isotc211.org/2005/gmd}westBoundLongitude).
        # Stripping everything before and including '}' allows us to search for simple tags like './/westBoundLongitude'.
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        # --- 1. Bounding Box Extraction ---
        # Locate the decimal text for each geographical boundary and cast to float.
        w = float(root.find('.//westBoundLongitude/Decimal').text)
        e = float(root.find('.//eastBoundLongitude/Decimal').text)
        s = float(root.find('.//southBoundLatitude/Decimal').text)
        n = float(root.find('.//northBoundLatitude/Decimal').text)
        
        # Package into standard STAC/GeoJSON bounding box array format: [minX, minY, maxX, maxY]
        bbox = [w, s, e, n]

        # --- 2. CRS (EPSG) Extraction ---
        try:
            # Locate the EPSG anchor tag (e.g., <Anchor>EPSG:3035</Anchor>)
            epsg_text = root.find('.//referenceSystemIdentifier//Anchor').text
            
            # Use regex to extract only the numeric portion (e.g., '3035' from 'EPSG:3035')
            epsg = int(re.search(r'\d+', epsg_text).group())
        except AttributeError:
            # Fallback to standard WGS84 if the CRS tag is missing or malformed
            epsg = 4326

        # --- 3. Spatial Resolution Extraction ---
        try:
            # Attempt to read the resolution directly from the <Distance> tag
            res_text = root.find('.//spatialResolution//Distance').text
            res = float(res_text)
            
            # Convert pure integer floats (e.g., 10.0) to clean integers (10) for schema consistency
            if res.is_integer(): 
                res = int(res)
                
        except (AttributeError, ValueError, TypeError):
            # Fallback: Copernicus metadata frequently contains typos where the tile ID (e.g., 'E9N27') 
            # is accidentally injected into the resolution tag. If float conversion fails, parse the filename.
            
            # Regex looks for the standard resolution block in the filename (e.g., '_R10m_' or '_R1km_')
            match = re.search(r'_R(\d+)(m|km)_', filename, re.IGNORECASE)
            
            if match:
                # Extract the numeric value and the unit (m or km)
                val = int(match.group(1))
                unit = match.group(2).lower()
                
                # Standardize everything to meters
                res = val * 1000 if unit == 'km' else val
            else:
                # Ultimate fallback if both the XML and the filename lack resolution data
                res = 10 

        # --- 4. Collection Metadata Extraction ---
        # Extract the dataset abstract. Safely check if the element exists before calling .text
        abstract_elem = root.find('.//abstract/CharacterString')
        abstract = abstract_elem.text if abstract_elem is not None else "Copernicus Dataset"
        
        # Extract the dataset lineage/processing history.
        lineage_elem = root.find('.//lineage//statement/CharacterString')
        lineage = lineage_elem.text if lineage_elem is not None else ""

        return bbox, epsg, res, abstract, lineage
        
    except Exception:
        # Global catch-all to prevent pipeline crashes on entirely malformed XML files.
        # Returns safe defaults that allow STAC Item generation to proceed.
        return None, 4326, 10, "Copernicus Dataset", ""

import os
import re
import json
from datetime import datetime
import geopandas as gpd
from shapely.geometry import box

def build_geoparquet_catalog(
    base_prefix, 
    output_dir="./geoparquet_catalog",
    dataset_name="Unknown Dataset",
    variable_name="Unknown Variable",
    raster_asset_name="raster",
    metadata_asset_name="metadata_xml",
    aux_asset_name="aux_xml",
    tile_code_regex=r'_R\d+(?:m|km)_([A-Z0-9]+)_',
    bucket_name="eodata"
):
    """
    Builds a spatially-indexed GeoParquet catalog from a structured S3 dataset.

    This orchestrator function crawls a designated S3 prefix to discover 
    geospatial assets, dynamically extracts standard ISO metadata, GDAL 
    auxiliary statistics, and styling legends, and compiles the structured 
    records into a single GeoPandas dataframe. It generalizes the extraction 
    process by accepting variable names, dataset identifiers, and configurable 
    asset keys. The resulting dataframe is serialized to a GeoParquet file.

    Parameters
    ----------
    base_prefix : str
        The root S3 prefix (folder path) where the crawler should begin.
    output_dir : str, optional
        The local directory path where the generated `catalog.parquet` and 
        `legend.json` files will be saved. Defaults to "./geoparquet_catalog".
    dataset_name : str, optional
        A human-readable name identifying the dataset to store in the catalog. 
        Defaults to "Unknown Dataset".
    variable_name : str, optional
        The specific variable or observation type this catalog represents. 
        Defaults to "Unknown Variable".
    raster_asset_name : str, optional
        The dictionary key used by the crawler for the main image raster. 
        Defaults to "raster".
    metadata_asset_name : str, optional
        The dictionary key used by the crawler for the ISO metadata XML file. 
        Defaults to "metadata_xml".
    aux_asset_name : str, optional
        The dictionary key used by the crawler for the GDAL auxiliary statistics 
        file. Defaults to "aux_xml".
    tile_code_regex : str, optional
        A regular expression pattern used to extract the specific geographic 
        tile code from the tile's filename. Defaults to a pattern tailored for 
        Copernicus data (`r'_R\d+(?:m|km)_([A-Z0-9]+)_'`). Pass None to skip.
    bucket_name : str, optional
        The S3 bucket name containing the target assets. Defaults to "eodata".

    Returns
    -------
    None
        The function does not return a value. It writes the resulting 
        `.parquet` and `.json` files directly to the specified `output_dir`.

    Notes
    -----
    - **Global Dependencies:** Requires the globally instantiated `s3_client` 
      and previously defined parsing functions (`discover_s3_assets_bfs`, 
      `parse_legend_colors`, `parse_iso_metadata`, `parse_aux_xml_stats`).
    - **Schema Constraints:** Because Parquet enforces strict column schemas, 
      nested dynamic dictionaries (like pixel statistics) are serialized into 
      JSON strings before being added to the dataframe.
    - **Date Extraction:** Temporal metadata is extracted dynamically from 
      the folder structure (e.g., matching `YYYY/MM/DD` or `YYYY`).

    Examples
    --------
    >>> target_folder = "CLMS/landcover_landuse/crop_types/.../v1/"
    >>> build_geoparquet_catalog(
    ...     base_prefix=target_folder,
    ...     dataset_name="Copernicus Crop Types",
    ...     variable_name="Crop Classifications"
    ... )
    Discovered 7216 potential data folders.
    Exported legend.json
    Extracting metadata and building GeoParquet records...
    Converting 7216 items to GeoDataFrame...
    Successfully saved GeoParquet Catalog: /absolute/path/to/geoparquet_catalog/catalog.parquet
    """
    # Ensure the target output directory exists before attempting to write files
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Crawl S3 ---
    # Utilize the breadth-first search crawler to safely map the bucket hierarchy
    # and group related assets (raster, xml, aux.xml) by their parent folder.
    tiles, legend_s3_key = discover_s3_assets_bfs(base_prefix)
    print(f"Discovered {len(tiles)} potential data folders.")

    # --- 2. Extract Legend ---
    global_legend = {}
    if legend_s3_key:
        try:
            # Fetch the QML/SLD style file from S3 into memory
            leg_obj = s3_client.get_object(Bucket=bucket_name, Key=legend_s3_key)
            
            # Parse the XML to extract the official class mapping (Code -> Name/Color)
            global_legend = parse_legend_colors(leg_obj['Body'].read())
            
            # Save the parsed legend as a standalone JSON file alongside the catalog.
            # This allows rendering/visualization scripts to load colors instantly.
            with open(os.path.join(output_dir, "legend.json"), "w") as f:
                json.dump(global_legend, f, indent=4)
            print("Exported legend.json")
        except Exception as e:
            print(f"Failed to extract legend: {e}")

    # --- 3. Process Tiles into STAC Items ---
    records = []
    
    print("Extracting metadata and building GeoParquet records...")
    for folder, assets in tiles.items():
        # Skip incomplete asset groups that lack the core raster or metadata
        if raster_asset_name not in assets or metadata_asset_name not in assets:
            continue

        # Extract Date dynamically from the folder path using regex.
        # Checks for an explicit YYYY/MM/DD structure first.
        date_match = re.search(r'/([12]\d{3})/(\d{2})/(\d{2})/', folder)
        if date_match:
            item_date = datetime(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
        else:
            # Fallback to matching just the year (YYYY), defaulting to Jan 1st.
            year_match = re.search(r'/([12]\d{3})/', folder)
            item_date = datetime(int(year_match.group(1)), 1, 1) if year_match else datetime(2017, 1, 1)

        # Fetch the ISO metadata XML file for this specific tile
        xml_obj = s3_client.get_object(Bucket=bucket_name, Key=assets[metadata_asset_name])
        
        # Pass both the XML content and the base folder name (the Tile ID) to the parser.
        # The filename acts as a fallback for extracting resolution if the XML is malformed.
        tile_id = os.path.basename(folder)
        bbox, epsg, res, global_abstract, _ = parse_iso_metadata(xml_obj['Body'].read(), filename=tile_id)
        
        # If bounding box extraction failed entirely, the record cannot be spatialized; skip it.
        if not bbox: 
            continue

        # Parse Aux Stats if the GDAL auxiliary file is present in the S3 directory
        stats = {}
        if aux_asset_name in assets:
            aux_obj = s3_client.get_object(Bucket=bucket_name, Key=assets[aux_asset_name])
            stats = parse_aux_xml_stats(aux_obj['Body'].read())

        # Extract the specific Tile Code from the filename/folder name.
        # Uses the configurable regular expression provided in the function arguments.
        tile_code = "unknown"
        if tile_code_regex:
            tile_code_match = re.search(tile_code_regex, tile_id)
            if tile_code_match:
                tile_code = tile_code_match.group(1)

        # Construct a flattened dictionary representing one row in the Parquet file.
        record = {
            "id": tile_id,
            "dataset": dataset_name,
            "variable": variable_name,
            "tile_code": tile_code,
            # Generate a Shapely Polygon from the BBox for GeoPandas spatial indexing
            "geometry": box(bbox[0], bbox[1], bbox[2], bbox[3]),
            "datetime": item_date,
            # Explode the bounding box into explicit numerical columns for fast scalar filtering
            "bbox_west": bbox[0], "bbox_south": bbox[1], "bbox_east": bbox[2], "bbox_north": bbox[3],
            "proj:epsg": epsg,
            "proj:resolution": res,
            "s3_href": f"s3://{bucket_name}/{assets[raster_asset_name]}",
            # Parquet requires uniform column types. A dictionary with varying keys breaks the schema.
            # We serialize the stats dictionary into a JSON string to store it safely in a single String column.
            "statistics": json.dumps(stats)
        }
        records.append(record)

    # Abort if no valid records were processed
    if not records:
        print("No valid items found to build the catalog.")
        return

    # --- 4. Generate GeoParquet ---
    print(f"Converting {len(records)} items to GeoDataFrame...")
    
    # Initialize the GeoDataFrame, assigning the geometry column and declaring the base CRS (WGS84)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    
    # Write directly to Parquet format (index=False prevents saving the arbitrary pandas row numbers)
    parquet_path = os.path.join(output_dir, "catalog.parquet")
    gdf.to_parquet(parquet_path, index=False)
    
    print(f"Successfully saved GeoParquet Catalog: {os.path.abspath(parquet_path)}")

