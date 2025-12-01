import numpy as np


def compute_pixel_lifetime(image_file):
    """Method to compute the protein lifetime from the pulse and
    chase image channels

    INPUT PARAMETERS
        image_file - 3d np.array of image file with hight x width x channels

    OUTPUT PARAMETERS
        image_output - 3d np.array with pulse, chase, and lifetime channels


    """
    # Constants for calibration
    PULSE_CORRECTION = 67.7
    PULSE_SLOPE = 1169.1
    CHASE_CORRECTION = 67.7
    CHASE_SLOPE = 1169.1
    TIME_CONSTANT = 3
    eps = 1e-9  # small num to prevent inf

    # Seperate the image channels
    chase_image = image_file[:, :, 2]
    pulse_image = image_file[:, :, 3]

    # Normalize the images to concentration values
    chase_norm = normalize_image(chase_image, CHASE_CORRECTION, CHASE_SLOPE)
    pulse_norm = normalize_image(pulse_image, PULSE_CORRECTION, PULSE_SLOPE)

    # Get rid of negative values in empty space
    chase_norm[chase_norm < 0] = 0
    pulse_norm[pulse_norm < 0] = 0

    # Calculate the fraction ratio
    total_protein = chase_norm + pulse_norm
    fraction_ratio = pulse_norm + total_protein

    # Add small value to ensure no division by zero error
    fraction_ratio_corrected = fraction_ratio + eps

    # Calculate the lifetime
    lifetime = np.absolute(TIME_CONSTANT / (np.log(1 / fraction_ratio_corrected)))

    image_output = np.dstack([pulse_image, chase_image, lifetime])

    return image_output


def normalize_image(image, correction, slope):
    """Helper function to normalize images"""

    # Correct for baseline fluorescence
    corrected_image = image - correction
    normalized_image = corrected_image / slope

    return normalized_image
