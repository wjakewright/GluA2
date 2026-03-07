import os
from collections import defaultdict
import json
import numpy as np
from skimage import io as sio
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from compute_pixel_lifetime import compute_pixel_lifetime
import qupath_utils as qutils


def export_normalized_images(mouse):
    """"""
    # Paths
    initial_path = r"Z:\People\Jake\Histology\GluA2"
    mouse_path = os.path.join(initial_path, mouse)
    fname_path = os.path.join(mouse_path, "abbaProject", "atlas_json")
    load_path = os.path.join(mouse_path, "QProject", "exported_tiffs")
    save_path = os.path.join(mouse_path, "normalized_images")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Load the image in
    fnames = next(os.walk(fname_path))[2]
    base_names = [x.split(".")[0] for x in fnames]

    for i, file_name in enumerate(base_names):
        image_file = file_name + ".tif"
        json_file = fnames[i]

        current_image = sio.imread(
            os.path.join(load_path, image_file), plugin="tifffile"
        )
        current_atlas = json.load(open(os.path.join(fname_path, json_file), "r"))

        dapi_image = current_image[:, :, 0]

        norm_image = compute_pixel_lifetime(current_image)

        chase_norm = norm_image[:, :, 1]
        pulse_norm = norm_image[:, :, 0]
        lifetime_image = norm_image[:, :, 2]

        chase_name = "chase_" + file_name + ".tif"
        chase_path = os.path.join(save_path, chase_name)
        pulse_name = "pulse_" + file_name + ".tif"
        pulse_path = os.path.join(save_path, pulse_name)
        dapi_name = "dapi_" + file_name + ".tif"
        dapi_path = os.path.join(save_path, dapi_name)
        sio.imsave(chase_path, chase_norm)
        sio.imsave(pulse_path, pulse_norm)
        sio.imsave(dapi_path, dapi_image)

        # get the masks
        masks = get_slice_masks(lifetime_image, current_atlas)

        # Construct a figure
        cmap = cm.get_cmap("viridis")
        cmap.set_bad(color="black")

        plot_image = lifetime_image * masks[0]
        plot_image[np.isnan(plot_image)] = 0
        plot_image[plot_image == 0] = np.nan

        plt.figure(figsize=(10, 10))
        im = plt.imshow(plot_image, cmap=cmap, vmin=0, vmax=5)
        plt.colorbar(im, label="Lifetime tau (d)")

        for mask in masks:
            contours = measure.find_contours(mask, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color="white", linewidth=0.4)
        plot_name = os.path.join(save_path, f"lifetime_{file_name}.pdf")
        plt.savefig(plot_name)


def get_slice_masks(image_data, json_file):
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
    masks = []
    # Iterate through each feature seperately
    for feature in features:

        # Store name and feature relationships
        # fix clip of the roi geometry
        geometry = qutils.fix_clip(feature["geometry"], H=H, W=W)
        # Extract the pixel lifetimes
        ## Move on if roi is empty
        if geometry is None:
            continue

        ## Get mask
        mask = qutils.mask_from_geom(geometry, H, W)
        masks.append(mask)

    return masks
