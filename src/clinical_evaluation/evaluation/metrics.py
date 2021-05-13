import SimpleITK as sitk
import numpy as np
from typing import Optional

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


METRIC_DICT = {
    "ssim": lambda x, y, mask: ssim(*apply_mask(x, y, mask)),
    "mse": lambda x, y, mask: mse(*apply_mask(x, y, mask)),
    "nmse": lambda x, y, mask: nmse(*apply_mask(x, y, mask)),
    "psnr": lambda x, y, mask: psnr(*apply_mask(x, y, mask)),
    "mae": lambda x, y, mask: mae(*apply_mask(x, y, mask))
}


def calculate_metrics(target, pred, mask=None, limit=(-1000, 2000)):
    target = sitk.GetArrayFromImage(target)
    pred = sitk.GetArrayFromImage(pred)

    target = np.clip(target, *limit)
    pred = np.clip(pred, *limit)

    target, pred = target - limit[0], pred - limit[0]

    metrics = {}
    for label, metric_function in METRIC_DICT.items():
        metrics[label] = metric_function(target, pred, mask)

    return metrics


def apply_mask(gt, pred, mask=None):

    if mask is not None:
        mask = sitk.GetArrayFromImage(mask).astype(np.bool)
        negated_mask = ~mask
        gt = np.ma.masked_array(gt*mask, mask=negated_mask)
        pred = np.ma.masked_array(pred*mask, mask=negated_mask)

    return gt, pred


# Metrics below are taken from
# https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Added MAE to the list of metrics, change SSIM to 3D implementation

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


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]

def relative_difference(gt: np.ndarray,  pred: np.ndarray, type='percent_error'):
    #https://en.wikipedia.org/wiki/Relative_change_and_difference
    # Since averaging is done, it gives percent error in terms of absolutes.
    # No information about over/under estimation is provided.
    return np.mean(np.abs(gt - pred)/ np.abs(gt))
