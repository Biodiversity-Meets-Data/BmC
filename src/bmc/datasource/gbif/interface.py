import os
import itertools
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pygbif
import requests
@lru_cache(maxsize=None)
def lookup_backbone_single(
    name_query: str, 
    dataset_uuid: str | None = None
) -> dict:
    """Execute a cached single-name taxonomic lookup against GBIF or an alternative backbone.

    Queries either the standard GBIF Backbone Taxonomy or a specific external 
    checklist hosted on GBIF. Normalizes and flattens the variations in API 
    payload responses into a standardized dictionary schema.

    Parameters
    ----------
    name_query : str
        The scientific or taxonomic name string to resolve.
    dataset_uuid : str, optional
        The GBIF dataset UUID key assigned to a target checklist taxonomy. 
        If None, uses the official strict GBIF backbone matching engine.

    Returns
    -------
    dict
        A flattened dictionary mapping unified taxonomic keys. Essential keys 
        include: 'matchType', 'usageKey', 'acceptedUsageKey', 'scientificName', 
        'rank', 'status', and 'lookupNames'.
    """
    if not name_query:
        return {"matchType": "NONE", "lookupNames": name_query}
        
    try:
        if dataset_uuid:
            # --- ALTERNATIVE DATASET LOOKUP ---
            res = pygbif.species.name_lookup(q=name_query, datasetKey=dataset_uuid, limit=50)
            results = res.get('results', [])
            
            if not results:
                return {"matchType": "NONE", "note": "No match in dataset", "lookupNames": name_query}
                
            exact_matches = []
            for r in results:
                canon = r.get('canonicalName', r.get('scientificName', '')).lower()
                sci = r.get('scientificName', '').lower()
                if name_query.lower() in [canon, sci]:
                    exact_matches.append(r)
                    
            if exact_matches:
                best = next((r for r in exact_matches if r.get('taxonomicStatus') == 'ACCEPTED'), exact_matches[0])
                match_type = "EXACT"
            else:
                best = results[0]
                match_type = "FUZZY"
                
            raw_usage = best.get("taxonID", best.get("key"))
            raw_accepted = best.get("acceptedNameUsageID", best.get("acceptedKey", raw_usage))
            
            usage_key = int(raw_usage) if str(raw_usage).isdigit() else raw_usage
            accepted_key = int(raw_accepted) if str(raw_accepted).isdigit() else raw_accepted
                
            return {
                "matchType": match_type,
                "usageKey": usage_key,
                "acceptedUsageKey": accepted_key,
                "scientificName": best.get("scientificName"),
                "rank": best.get("rank"),
                "status": best.get("taxonomicStatus"),
                "lookupNames": name_query
            }
            
        else:
            # --- STANDARD GBIF LOOKUP ---
            res = pygbif.species.name_backbone(name_query, strict=False)
            
            if not res:
                return {"matchType": "NONE", "note": "Empty GBIF response", "lookupNames": name_query}
            
            # Extract Match Type safely
            if "diagnostics" in res:
                match_type = res["diagnostics"].get("matchType", "NONE")
            else:
                match_type = res.get("matchType", "NONE")
                
            if match_type == "NONE":
                return {"matchType": "NONE", "note": "No GBIF match", "lookupNames": name_query}
                
            usage = res.get("usage", res)
            accepted = res.get("acceptedUsage", usage)
            
            raw_usage_key = usage.get("key", res.get("usageKey"))
            raw_accepted_key = accepted.get("key", res.get("acceptedUsageKey", res.get("acceptedKey")))
            
            # SAFE CASTING: Check if it's a digit first, otherwise keep it as a string
            usage_key = int(raw_usage_key) if raw_usage_key is not None and str(raw_usage_key).isdigit() else raw_usage_key
            accepted_key = int(raw_accepted_key) if raw_accepted_key is not None and str(raw_accepted_key).isdigit() else raw_accepted_key
            
            return {
                "matchType": match_type,
                "usageKey": usage_key,
                "acceptedUsageKey": accepted_key,
                "scientificName": usage.get("name", res.get("scientificName")),
                "rank": usage.get("rank", res.get("rank")),
                "status": usage.get("status", res.get("status")),
                "lookupNames": name_query
            }
            
    except Exception as e:
        return {"matchType": "NONE", "note": f"API Error: {e}", "lookupNames": name_query}

@lru_cache(maxsize=None)
def lookup_usage_single(taxon_key: int | str, 
                        dataset_uuid: str | None = '7ddf754f-d193-4cc9-b351-99906754a03b') -> dict | None:
    """Execute a cached single-key GBIF or alternative dataset name usage API lookup.

    Parameters
    ----------
    taxon_key : int or str
        The raw unique identifier.
    dataset_uuid : str, optional
        The target dataset key to search. Defaults to the Catalogue of Life 
        dataset UUID ('7ddf754f-d193-4cc9-b351-99906754a03b'). If explicitly 
        set to None, uses the standard GBIF backbone.

    Returns
    -------
    dict or None
        The parsed JSON dictionary response containing taxonomic metadata if successful.
    """
    try:
        if dataset_uuid:
            res = pygbif.species.name_usage(datasetKey=dataset_uuid, sourceId=str(taxon_key))
            if res and "results" in res and len(res["results"]) > 0:
                return res["results"][0]
            return None
        else:
            # Safely cast to int only if it is a number
            safe_key = int(taxon_key) if str(taxon_key).isdigit() else taxon_key
            return pygbif.species.name_usage(key=safe_key)
    except Exception:
        return None

def lookup_backbone(
    name_list: list[str], 
    max_workers: int = 8, 
    dataset_uuid: str | None = None
) -> list[dict]:
    """Execute high-speed parallel backbone lookups across an array of names."""
    lookup_func = partial(lookup_backbone_single, dataset_uuid=dataset_uuid)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lookup_func, name_list))


def fetch_taxon_info(
    species_names: list[str], 
    out_file: str = "", 
    out_path: str = "", 
    mismatch_file: str = "", 
    keep_higherrank: bool = False,
    keep_fuzzy: bool = False,
    max_workers: int = 8,
    dataset_uuid: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a list of taxonomic names and partition them by query accuracy."""
    species_names = list(set(name.strip() for name in species_names if name and name.strip()))
    if not species_names:
        return pd.DataFrame(), pd.DataFrame()
    
    taxonomic_info = lookup_backbone(
        species_names, 
        max_workers=max_workers, 
        dataset_uuid=dataset_uuid
    )
    taxonomic_df = pd.DataFrame(taxonomic_info)
    
    if "lookupNames" not in taxonomic_df.columns:
        taxonomic_df["lookupNames"] = species_names

    mismatchTypes = [
        "NONE",
        *(["FUZZY", "VARIANT"] if not keep_fuzzy else []),
        *(["HIGHERRANK"] if not keep_higherrank else [])
    ]
                     
    if "matchType" not in taxonomic_df.columns:
        taxonomic_df["matchType"] = "NONE"
        
    mismatch_indices = taxonomic_df["matchType"].isin(mismatchTypes)
    
    if set(mismatch_indices) == {False, True}:
        counts = "".join(f"\nCount {mt}: {sum(taxonomic_df['matchType'] == mt)}" for mt in set(taxonomic_df['matchType']))
        print(f"Mismatches encountered of type {set(taxonomic_df['matchType'])}{counts}")
        
    mismatch_df = taxonomic_df[mismatch_indices].copy()
    taxonomic_df = taxonomic_df[~mismatch_indices].copy()

    if not taxonomic_df.empty:
        if "acceptedUsageKey" not in taxonomic_df.columns:
            taxonomic_df["acceptedUsageKey"] = taxonomic_df.get("usageKey", None)
        else:
            taxonomic_df["acceptedUsageKey"] = taxonomic_df["acceptedUsageKey"].fillna(taxonomic_df.get("usageKey"))
            
        if dataset_uuid:
            taxonomic_df["acceptedUsageKey"] = taxonomic_df["acceptedUsageKey"].astype(str)
        else:
            # Safely cast to int only if it's a number, leaving alphanumeric strings (or NaNs) as-is
            taxonomic_df["acceptedUsageKey"] = taxonomic_df["acceptedUsageKey"].apply(
                lambda x: int(x) if str(x).isdigit() else x
            )

    if out_file != "":
        taxonomic_df.to_csv(os.path.join(out_path, out_file), index=False)
        
    if mismatch_file != "" and not mismatch_df.empty:
        cols_to_save = [col for col in ["matchType", "scientificName", "lookupNames"] if col in mismatch_df.columns]
        mismatch_df[cols_to_save].to_csv(os.path.join(out_path, mismatch_file), index=False)
        
    return taxonomic_df, mismatch_df

def match_names_to_keys(names: list[str], 
                        max_workers: int = 8,
                        dataset_uuid: str | None = None) -> list[int | str | None]:
    """Resolve a list of taxonomic scientific names to their unique keys."""
    if not names:
        return []
        
    unique_names = list(set(n for n in names if n))
    lookup_func = partial(lookup_backbone_single, dataset_uuid=dataset_uuid)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [result.get("usageKey") for result in executor.map(lookup_func, unique_names)]
        
    return results