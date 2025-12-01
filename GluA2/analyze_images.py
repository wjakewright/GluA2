import os
from collections import defaultdict

import json, re
import numpy as np
import pandas as pd
import tifffile

from skimage import io as sio

import qupath_utils as qutils
from organize_roi_features import extract_roi_values
from compute_pixel_lifetime import compute_pixel_lifetime


def analyze_images(
    mouse_list,
    save=True,
):
    """Function to handle the preprocessing of aligned images and extract pixels for
    the different atlas ROIs

    INPUT PARAMETERS
        mouse_list - list of str specifying the mice to be analyzed

        save - boolean specifying whether to save the output of the data

    """
    initial_path = r"Z:\People\Jake\Histology\GluA2"

    # Iterate through each mouse seperately
    for mouse in mouse_list:
        # Setup the paths to load the data from
        mouse_path = os.path.join(initial_path, mouse)
        image_file_path = os.path.join(mouse_path, "QProject", "exported_tiffs")
        json_file_path = os.path.join(mouse_path, "abbaProject", "atlas_json")

        # Get all of the json file names
        json_fnames = next(os.walk(json_file_path))[2]
        base_fnames = [x.split(".")[0] for x in json_fnames]

        # Iterate through each image file seperately
        ## Set up temporary variable to store each slice
        slice_data = []

        for i, file_name in enumerate(base_fnames):
            image_file = file_name + ".tif"
            json_file = json_fnames[i]

            # Load in the image
            current_image = sio.imread(
                os.path.join(image_file_path, image_file), plugin="tifffile"
            )
            # Load in the atlas
            current_atlas = json.load(
                open(os.path.join(json_file_path, json_file), "r")
            )

            # Compute the pixel-wise protein lifetime
            lifetime_image = compute_pixel_lifetime(current_image)

            # Extract lifetime for each ROI
            roi_df = extract_roi_values(lifetime_image, current_atlas)

            # Store the data
            slice_data.append(roi_df)

        # Concatenate dataframes
        concatenated_df = pd.concat(slice_data, ignore_index=True)
        # Aggregate data within brain regions
        combined_df = concatenated_df.groupby(
            by=["level", "roi_name", "side", "parent_name", "orig_id", "parent_id"],
            as_index=False,
        ).agg(
            tot_pixels=("n_pixels", "sum"),
            mean_pulse=("mean_pulse", "mean"),
            mean_chase=("mean_chase", "mean"),
            mean_lifetime=("mean_lifetime", "mean"),
        )

        # Save the data
        if save:
            save_path = os.path.join(mouse_path, "analyzed_data")
            save_name = os.path.join(save_path, f"{mouse}_analyzed_data.pickle")
            combined_df.to_pickle(save_name)
