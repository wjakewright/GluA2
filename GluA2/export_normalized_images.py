import os
import numpy as np
from skimage import io as sio


def export_normalized_images(mouse):
    """"""
    # Constants for calibration
    PULSE_CORRECTION = 66.68
    PULSE_SLOPE = 480.47
    CHASE_CORRECTION = 67.7
    CHASE_SLOPE = 1169.1
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

        current_image = sio.imread(
            os.path.join(load_path, image_file), plugin="tifffile"
        )
        chase_image = current_image[:, :, 1]
        pulse_image = current_image[:, :, 0]

        chase_norm = normalize_image(chase_image, CHASE_CORRECTION, CHASE_SLOPE)
        pulse_norm = normalize_image(pulse_image, PULSE_CORRECTION, PULSE_SLOPE)

        chase_name = "chase_" + file_name + ".tif"
        chase_path = os.path.join(save_path, chase_name)
        pulse_name = "pulse_" + file_name + ".tif"
        pulse_path = os.path.join(save_path, pulse_name)

        sio.imsave(chase_path, chase_norm)
        sio.imsave(pulse_path, pulse_norm)


def normalize_image(image, correction, slope):
    """Helper function to normalize images"""

    # Correct for baseline fluorescence
    corrected_image = image - correction
    normalized_image = corrected_image / slope

    return normalized_image
