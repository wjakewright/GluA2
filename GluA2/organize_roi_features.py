from collections import defaultdict

import pandas as pd
import numpy as np
import qupath_utils as qutils
from scipy.ndimage import median_filter


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
            # mean_lifetime = calculate_mean_lifetime(
            #     image_data[:, :, 0], image_data[:, :, 1], mask
            # )

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


def extract_roi_pixel_values(image_data, json_file, mouse_id, modality):
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
    id_to_name = {}
    id_to_feat = {}
    # Iterate through each feature selectively
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

        # fix clip of the roi geometry
        geometry = qutils.fix_clip(feature["geometry"], H=H, W=W)
        # Extract the pixel lifetimes
        ## Move on if roi is empty
        if geometry is None:
            record_dict = {
                "mouse_id": mouse_id,
                "roi_name": name,
                "side": side,
                "mean_lifetime": np.nan,
                "orig_id": orig_id,
                "parent_id": parent_id,
            }
            records.append(record_dict)
            continue

        ## Get mask
        mask = qutils.mask_from_geom(geometry, H, W)
        ## Pull the lifetime values from the mask
        if not mask.any():
            mean_lifetime = np.nan
            record_dict = {
                "mouse_id": np.array([mouse_id for _ in len(mean_lifetime)]),
                "roi_name": np.array([name for _ in len(mean_lifetime)]),
                "side": [side for _ in len(mean_lifetime)],
                "mean_lifetime": mean_lifetime,
                "orig_id": [orig_id for _ in len(mean_lifetime)],
                "parent_id": [parent_id for _ in len(mean_lifetime)],
            }
            records.append(record_dict)
            continue
        else:
            if modality == "lifetime":
                # mean_lifetime = get_all_masked_pixels(image_data[:, :, 2], mask)
                mean_lifetime = calculate_pixel_lifetime(
                    image_data[:, :, 0], image_data[:, :, 1], mask
                )
            elif modality == "pulse":
                mean_lifetime = get_all_masked_pixels(image_data[:, :, 0], mask)
            elif modality == "chase":
                mean_lifetime = get_all_masked_pixels(image_data[:, :, 1], mask)

        # Store the values
        record_dict = {
            "mouse_id": np.array([mouse_id for _ in range(len(mean_lifetime))]),
            "roi_name": np.array([name for _ in range(len(mean_lifetime))]),
            "side": [side for _ in range(len(mean_lifetime))],
            "mean_lifetime": mean_lifetime,
            "orig_id": [orig_id for _ in range(len(mean_lifetime))],
            "parent_id": [parent_id for _ in range(len(mean_lifetime))],
        }
        records.append(record_dict)

    # Concatenate dictionaries into a single large one
    concate_records = defaultdict(list)
    for r in records:
        for k, v in r.items():
            concate_records[k].extend(v)

    # Transform into dataframe
    roi_df = pd.DataFrame.from_dict(concate_records)

    # Add parent name
    roi_df["parent_name"] = roi_df.apply(lambda x: x["parent_id"], axis=1)

    # Tidy up the dataframe
    ## Exclusions
    roi_df = qutils.drop_branches(
        roi_df, superparent_label="fiber tracts", recursive=True
    )
    roi_df = qutils.drop_branches(roi_df, superparent_label="VS", recursive=True)

    roi_df = roi_df[~(roi_df["roi_name"] == "Root")]

    # Drop nan values
    roi_df = roi_df.dropna(axis=0, how="any")
    roi_df = roi_df.drop(labels=["parent_id", "parent_name", "orig_id"], axis=1)

    roi_df = roi_df.sort_values(["roi_name", "side"]).reset_index(drop=True)

    return roi_df


def calculate_mean_lifetime(pulse, chase, mask):
    eps = 1e-9
    pulse_image = pulse[mask].flatten()
    chase_image = chase[mask].flatten()

    total = chase_image + pulse_image
    fraction_ratio = pulse_image + total
    fraction_ratio = fraction_ratio + eps
    fraction = pulse_image / fraction_ratio
    fraction = fraction[np.isfinite(fraction)] + eps

    lifetime = np.absolute(3 / (np.log(1 / fraction)))

    return np.nanmean(lifetime)


def calculate_pixel_lifetime(pulse, chase, mask):
    MAX_NUM = 1000
    pulse_image = pulse[mask].flatten()
    chase_image = chase[mask].flatten()

    total = chase_image + pulse_image
    fraction_ratio = pulse_image + total
    fraction_ratio = fraction_ratio[fraction_ratio != 0]
    fraction = pulse_image / fraction_ratio
    fraction = fraction[np.isfinite(fraction)]

    fraction[fraction == 0] = np.nan

    lifetime = np.absolute(3 / (np.log(1 / fraction)))

    lifetime = lifetime[~np.isnan(lifetime)]

    size = int(len(lifetime) / MAX_NUM)
    if size % 2 == 0:
        size = size + 1

    lifetime = median_filter(lifetime, size=size, mode="reflect")
    lifetime = lifetime[::size]

    return lifetime


def get_mean_masked_pixels(image, mask):
    masked_image = image[mask]
    all_pixels = masked_image.flatten()
    mean_pixels = np.nanmean(all_pixels)

    return mean_pixels


def get_all_masked_pixels(image, mask):
    MAX_NUM = 1000
    masked_image = image[mask]
    all_pixels = masked_image.flatten()
    size = int(len(all_pixels) / MAX_NUM)
    if size % 2 == 0:
        size = size + 1
    all_pixels = median_filter(all_pixels, size=size, mode="reflect")
    all_pixels = all_pixels[::size]

    return all_pixels
