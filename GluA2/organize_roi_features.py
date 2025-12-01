from collections import defaultdict

import pandas as pd
import numpy as np
import qupath_utils as qutils


def extract_features(json_file):
    """Function to organize the roi features from the geojson file

    INPUT PARAMETERS
        json_file - geojson file

    OUTPUT PARAMETERS
        meta_df - pd.dataframe of the roi information

    """
    features = [
        f
        for f in json_file.get("features", [])
        if isinstance(f, dict)
        and (f.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")
    ]

    # Initialize some dictionaries to store feature ids
    id_to_feat = {}
    parent_to_children = defaultdict(list)
    id_to_name = {}

    rows_meta = []

    # Iterate through each atlas feature
    for feature in features:
        properties = feature.get("properties") or {}
        measurements = properties.get("measurements") or {}
        orig_id = measurements.get("ID")
        parent_id = measurements.get("Parent ID")
        # Parse altas area name and hemisphere
        area, side = qutils.parse_area_side(properties)
        ## Get base area name and determine if it is a layer
        base = qutils.normalize_name(area)
        # Name map from ALL features
        if orig_id is not None:
            id_to_feat[orig_id] = feature
            id_to_name[orig_id] = base
        if parent_id is not None:
            parent_to_children[parent_id].append(orig_id)

        rows_meta.append(
            {
                "orig_id": orig_id,
                "parent_id": parent_id,
                "roi_name": base,
                "side": side,
            }
        )

    meta_df = pd.DataFrame(rows_meta)
    meta_df["orig_id"] = meta_df["orig_id"].astype("Int64")
    meta_df["parent_id"] = meta_df["parent_id"].astype("Int64")

    all_ids = set(id_to_feat.keys())
    parent_ids = set(parent_to_children.keys())
    leaf_ids = all_ids - parent_ids

    return meta_df, all_ids, parent_ids, leaf_ids


def extract_roi_values(image_data, json_file):
    """"""
    H = image_data.shape[0]
    W = image_data.shape[1]
    # Extract all roi features
    features = [
        f
        for f in json_file.get("features", [])
        if isinstance(f, dict)
        and (f.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")
    ]

    # Organize the rois and extract image data
    records = []
    id_to_feat = {}
    parent_to_children = defaultdict(list)
    id_to_name = {}
    # Iterate through each feature seperately
    for feature in features:
        properties = feature.get("properties") or {}
        measurements = properties.get("measurements") or {}
        orig_id = measurements.get("ID")
        parent_id = measurements.get("Parent ID")
        # Parse altas area name and hemisphere
        area, side = qutils.parse_area_side(properties)
        ## Get base area name and determine if it is a layer
        name = qutils.normalize_name(area)
        # Store name and feature relationships
        if orig_id is not None:
            id_to_feat[orig_id] = feature
            id_to_name[orig_id] = name
        if parent_id is not None:
            parent_to_children[parent_id].append(orig_id)
        # fix clip of the roi geometry
        geometry = qutils.fix_clip(feature["geometry"], H=H, W=W)
        # Extract the pixel lifetimes
        ## Move on if roi is empty
        if geometry is None:
            record_dict = {
                "level": None,
                "roi_name": name,
                "side": side,
                "parent_name": None,
                "orig_id": orig_id,
                "parent_id": parent_id,
                "n_pixels": 0,
                "mean_pulse": 0.0,
                "mean_chase": 0.0,
                "mean_lifetime": np.nan,
            }
            records.append(record_dict)
            continue

        ## Get mask
        mask = qutils.mask_from_geom(geometry, H, W)
        ## Pull the lifetime values from the mask
        if not mask.any():
            n_pixels = 0
            mean_pulse = 0.0
            mean_chase = 0.0
            mean_lifetime = np.nan
        else:
            n_pixels = int(mask.sum())
            mean_pulse = get_mean_masked_pixels(image_data[:, :, 0], mask)
            mean_chase = get_mean_masked_pixels(image_data[:, :, 1], mask)
            mean_lifetime = get_mean_masked_pixels(image_data[:, :, 2], mask)

        # Store the values
        record_dict = {
            "level": None,
            "roi_name": name,
            "side": side,
            "parent_name": None,
            "orig_id": orig_id,
            "parent_id": parent_id,
            "n_pixels": n_pixels,
            "mean_pulse": mean_pulse,
            "mean_chase": mean_chase,
            "mean_lifetime": mean_lifetime,
        }
        records.append(record_dict)

    # Transform into dataframe
    roi_df = pd.DataFrame.from_records(records)

    # Do some formatting for the dataframe
    roi_df["orig_id"] = roi_df["orig_id"].astype("Int64")
    roi_df["parent_id"] = roi_df["parent_id"].astype("Int64")

    all_ids = set(id_to_feat.keys())
    parent_ids = set(parent_to_children.keys())
    leaf_ids = all_ids - parent_ids

    # Add in the level values
    for index, row in roi_df.iterrows():
        oid = row["orig_id"]
        level = "orphan" if oid is None else ("leaf" if oid in leaf_ids else "parent")
        roi_df["level"].values[index] = level
        parent_name = id_to_name.get(row["parent_id"])
        roi_df["parent_name"].values[index] = parent_name

    # Tidy up the dataframe
    ## Exclusions
    roi_df = qutils.drop_branches(
        roi_df, superparent_label="fiber tracts", recursive=True
    )
    roi_df = qutils.drop_branches(roi_df, superparent_label="VS", recursive=True)

    roi_df = roi_df[~(roi_df["roi_name"] == "Root")]
    roi_df = roi_df.sort_values(
        ["level", "parent_name", "roi_name", "side"]
    ).reset_index(drop=True)

    return roi_df


def get_mean_masked_pixels(image, mask):
    masked_image = image[mask]
    mean_pixels = np.nanmean(masked_image)

    return mean_pixels
