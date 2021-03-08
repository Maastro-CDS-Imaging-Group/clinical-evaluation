import SimpleITK as sitk
import numpy as np
from typing import Optional

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

METRIC_DICT = {
    "ssim": lambda x, y, mask: ssim(*sitk2npy(x, y, mask)),
    "mse": lambda x, y, mask: mse(*sitk2npy(x, y, mask)),
    "nmse": lambda x, y, mask: nmse(*sitk2npy(x, y, mask)),
    "psnr": lambda x, y, mask: psnr(*sitk2npy(x, y, mask)),
    "mae": lambda x, y, mask: mae(*sitk2npy(x, y, mask))
}


def calculate_metrics(target, deformed_image, mask=None, offset=None):
    if offset is not None:
        target, deformed_image = target + offset, deformed_image + offset

    metrics = {}
    for label, metric_function in METRIC_DICT.items():
        metrics[label] = metric_function(deformed_image, target, mask)

    return metrics


def sitk2npy(gt: sitk.Image, pred: sitk.Image, mask=None):
    gt = sitk.GetArrayFromImage(gt)
    pred = sitk.GetArrayFromImage(pred)

    if mask is not None:
        mask = sitk.GetArrayFromImage(mask).astype(np.bool)
        negated_mask = ~mask
        gt = np.ma.masked_array(gt*mask, mask=negated_mask)
        pred = np.ma.masked_array(pred*mask, mask=negated_mask)

    return gt, pred


# Metrics below are taken from
# https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Added MAE to the list of metrics

def mae(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Absolute Error (MAE)"""
    return np.mean(np.abs(gt - pred))

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred)**2)

def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2


def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval)

    return ssim / gt.shape[0]

def relative_difference(gt: np.ndarray,  pred: np.ndarray, type='percent_error'):
    #https://en.wikipedia.org/wiki/Relative_change_and_difference
    # Since averaging is done, it gives percent error in terms of absolutes.
    # No information about over/under estimation is provided.
    return np.mean(np.abs(gt - pred)/ np.abs(gt))
