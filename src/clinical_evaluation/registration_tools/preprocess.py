import SimpleITK as sitk

from scipy import ndimage
import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)

# https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def hu_correction(image: sitk.Image, cval=-1024):
    """
    Offset CBCT values with certain cval
    """
    image = image + cval
    return image


def resample_image_to_spacing(image: sitk.Image,
                              new_spacing: list,
                              default_value: int,
                              interpolator: str = 'linear'):
    """Resample an image to a new spacing.
    """
    assert interpolator in SITK_INTERPOLATOR_DICT, \
        (f"Interpolator '{interpolator}' not part of SimpleITK. "
         f"Please choose one of the following {list(SITK_INTERPOLATOR_DICT.keys())}.")

    assert image.GetDimension() == len(new_spacing), \
        (f"Input is {image.GetDimension()}-dimensional while "
         f"the new spacing is {len(new_spacing)}-dimensional.")

    interpolator = SITK_INTERPOLATOR_DICT[interpolator]
    spacing = image.GetSpacing()
    size = image.GetSize()
    new_size = [
        int(round(siz * spac / n_spac)) for siz, spac, n_spac in zip(size, spacing, new_spacing)
    ]
    return sitk.Resample(
        image,
        new_size,  # size
        sitk.Transform(),  # transform
        interpolator,  # interpolator
        image.GetOrigin(),  # outputOrigin
        new_spacing,  # outputSpacing
        image.GetDirection(),  # outputDirection
        default_value,  # defaultPixelValue
        image.GetPixelID())  # outputPixelType


def get_connected_components(binary_array: np.ndarray,
                             structuring_element: np.ndarray = None) -> np.ndarray:
    """
    Returns a label map with a unique integer label for each connected geometrical object in the given binary array.
    Integer labels of components start from 1. Background is 0.
    """
    connected_component_array, _ = ndimage.label(binary_array, \
                                                                        structure=structuring_element)
    return connected_component_array


def smooth_contour_points(contour: np.ndarray, radius: int = 3, sigma: int = 10) -> np.ndarray:
    """
    Function that smooths contour points using the approach from 
    https://stackoverflow.com/a/37536310
    
    Simple explanation: Convolve 1D gaussian filter over the points to smoothen the curve
    """
    # Contour length is the total number of points + extra points
    # to ensure circularity.
    contour_length = len(contour) + 2 * radius
    # Last group of points.
    offset = (len(contour) - radius)

    x_filtered, y_filtered = [], []

    for idx in range(contour_length):
        x_filtered.append(contour[(offset + idx) \
                                          % len(contour)][0][0])

        y_filtered.append(contour[(offset + idx) \
                                          % len(contour)][0][1])

    # Gaussian blur from opencv is basically applying gaussian convolution
    # filter over these points.
    x_smooth = cv2.GaussianBlur(np.array(x_filtered), (radius, 1), sigma)
    y_smooth = cv2.GaussianBlur(np.array(y_filtered), (radius, 1), sigma)

    # Add smoothened point for
    smooth_contours = []
    for idx, (x, y) in enumerate(zip(x_smooth, y_smooth)):
        if idx < len(contour) + radius:
            smooth_contours.append(np.array([x, y]))

    return np.array(smooth_contours)


def get_body_mask(image: np.ndarray, HU_threshold: int) -> np.ndarray:
    """
    Function that gets a mask around the patient body and returns a 3D bound

    Parameters
    -------------
    image: Numpy array to get the mask and bound from
    HU_threshold: Set threshold to binarize image


    Returns
    -------------
    body_mask: Numpy array with same shape as input image as a body mask
    bound: Bounds around the largest component in 3D. This is in
    the ((z_min, z_max), (y_min, y_max), (x_min, x_max)) format
    """

    binarized_image = np.uint8(image >= HU_threshold)

    body_mask = np.zeros(image.shape, dtype=np.uint8)

    connected_components = get_connected_components(binarized_image)

    # Get counts for each component in the connected component analysis
    label_counts = [
        np.sum(connected_components == label) for label in range(1,
                                                                 connected_components.max() + 1)
    ]
    max_label = np.argmax(label_counts) + 1

    # Image with largest component binary mask
    binarized_image = connected_components == max_label

    for z in range(binarized_image.shape[0]):

        binary_slice = np.uint8(binarized_image[z])

        # Find contours for each binary slice
        try:
            contours, _ = cv2.findContours(binary_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except BaseException:
            logger.debug(
                "OpenCV could not find contours: Most likely this is a completely black image")
            continue

        if len(contours) == 0:
            continue

        # Get the largest contour based on its area
        largest_contour = max(contours, key=cv2.contourArea)

        # Smooth contour so that surface irregularities are removed better
        smoothed_contour = smooth_contour_points(largest_contour)

        # Project the points onto the body_mask image, everything
        # inside the points is set to 1.
        cv2.drawContours(body_mask[z], [smoothed_contour], -1, 1, -1)

    return body_mask
