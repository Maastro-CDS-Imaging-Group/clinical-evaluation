import SimpleITK as sitk
import numpy as np
from typing import Optional

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

METRIC_DICT = {
    "SSIM": lambda x, y: ssim(*sitk2npy(x, y)),
    "MSE": lambda x, y: mse(*sitk2npy(x, y)),
    "NMSE": lambda x, y: nmse(*sitk2npy(x, y)),
    "PSNR": lambda x, y: psnr(*sitk2npy(x, y))
}


def calculate_metrics(target, deformed_image):
    print("-" * 20)
    print("Computed Metrics between target and deformed image")
    print("-" * 20)
    for metric in METRIC_DICT:
        print(f"{metric}: {METRIC_DICT[metric](target, deformed_image)}")


def sitk2npy(gt: sitk.Image, pred: sitk.Image):
    gt = sitk.GetArrayFromImage(gt)
    pred = sitk.GetArrayFromImage(pred)
    return gt, pred


# Metrics below are taken from
# https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
# Copyright (c) Facebook, Inc. and its affiliates.


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred)**2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2


def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt: np.ndarray,
         pred: np.ndarray,
         maxval: Optional[float] = None) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval)

    return ssim / gt.shape[0]


def get_abs_diff(source: sitk.Image,
                 target: sitk.Image,
                 mask: sitk.Image = None):

    if mask:
        MaskImageFilter = sitk.MaskImageFilter()
        source = MaskImageFilter.Execute(source, mask)
        target = MaskImageFilter.Execute(target, mask)

    SubtractImageFilter = sitk.SubtractImageFilter()
    difference_image = SubtractImageFilter.Execute(source, target)

    AbsImageFilter = sitk.AbsImageFilter()
    abs_difference_image = AbsImageFilter.Execute(difference_image)
    return abs_difference_image


def get_statistics(image: sitk.Image):
    StatisticsImageFilter = sitk.StatisticsImageFilter()
    StatisticsImageFilter.Execute(image)
    mean = StatisticsImageFilter.GetMean()
    variance = StatisticsImageFilter.GetVariance()
    max = StatisticsImageFilter.GetMaximum()
    min = StatisticsImageFilter.GetMinimum()

    print(f"----- REPORT --------\n Mean: {mean} \n \
            Max: {max} \n Min: {min} \n Variance: {variance}")
