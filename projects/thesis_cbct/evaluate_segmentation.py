"""
Register, Visualize and Analyze Registrations. 

Register using SimpleElastix or ITKv4
Visualize using SimpleITK Checkboard, Overlay with image and video outputs
Analyze using metrics for HU differences, SSIM, PSNR, MSE, RMSE

"""
import logging
from pathlib import Path
import multiprocessing

from medpy import metric

import numpy as np
import SimpleITK as sitk
from clinical_evaluation.utils import visualizer
from clinical_evaluation.evaluation import metrics
from clinical_evaluation.registration import CSVSaver

from tqdm import tqdm
from clinical_evaluation.utils.logging import setup_logging
from lungmask import mask as automated_mask

logger = logging.getLogger(__name__)

masks = ("Lung R", "Lung L")


def main(args):
    data_folder = args.dataset_path.resolve()
    output_dir = args.output_dir.resolve()
    reginfo_data = CSVSaver(outdir=output_dir, save_to='patient_metrics.csv')

    if args.filter_by:
        patient_folders = [folder for folder in data_folder.iterdir() if folder.stem in args.filter_by]
    else:
        patient_folders = list(data_folder.iterdir())

    patient_list = [(folder, {
        'output_dir': output_dir,
        'data_folder': data_folder
    }) for folder in patient_folders]

    if args.cores > 1:
        logger.info(f"Running in multiprocessing mode with cores: {args.cores}")
        with multiprocessing.Pool(processes=args.cores) as pool:
            metrics = pool.starmap(process_patient_folder, patient_list)
            for metric in metrics:
                reginfo_data.add_info(metric)
            reginfo_data.save_info()
    else:
        logger.info(f"Running in main process only")
        for folder, meta_dict in patient_list:
            metric = process_patient_folder(folder, meta_dict)
            reginfo_data.add_info(metric)
        reginfo_data.save_info()

    mean_df = reginfo_data.get_aggregate_dataframe()
    mean_df = mean_df.transpose()
    mean_df.to_csv(output_dir / "mean_test_metrics.csv")


def process_patient_folder(folder, meta_dict):
    logger.info(f"Processing folder: {folder}")
    translated = folder / "translated.nrrd"
    deformed = folder / "deformed.nrrd"
    original = folder / "target.nrrd"

    if not (translated.exists() and deformed.exists() and original.exists()):
        logger.warning(f"Skipping {folder} as all required nrrd files are not present")
        return

    outdir = meta_dict['output_dir'] / folder.stem
    outdir.mkdir(exist_ok=True, parents=True)

    CT = sitk.ReadImage(str(deformed))
    CBCT = sitk.ReadImage(str(original))
    sCT = sitk.ReadImage(str(translated))
    metric_dict = {}

    if masks:
        rt_masks = {}
        for mask in masks:
            if isinstance(mask, tuple):
                label = mask[0]
                for mask_stem in mask[1]:
                    mask_path = (folder / mask_stem).with_suffix('.nrrd')
                    if mask_path.exists():
                        break

            else:
                label = mask
                mask_path = (folder / mask).with_suffix(".nrrd")

            if mask_path.exists():
                rt_masks[label] = sitk.ReadImage(str(mask_path))

    logger.info(f"Loaded RT masks: {rt_masks}")
    CT_mask = automated_mask.apply(CT)
    CBCT_mask = automated_mask.apply(CBCT)
    sCT_mask = automated_mask.apply(sCT)

    for idx, mask in enumerate(rt_masks):
        print(idx, mask)
        mask_indexed_CT = np.where(CT_mask==(idx+1), 1, 0)
        mask_indexed_CBCT = np.where(CBCT_mask==(idx+1), 1, 0)
        mask_indexed_sCT = np.where(sCT_mask==(idx+1), 1, 0)
        gt_mask_array = sitk.GetArrayFromImage(rt_masks[mask])

        metric_dict[f"{mask} CT"] = metric.binary.dc(mask_indexed_CT, gt_mask_array)
        metric_dict[f"{mask} CBCT"] = metric.binary.dc(mask_indexed_CBCT, gt_mask_array)
        metric_dict[f"{mask} sCT"] = metric.binary.dc(mask_indexed_sCT, gt_mask_array)

    metric_dict["save_dir"] = str(meta_dict['data_folder'])
    metric_dict["Patient"] = folder.stem
    return metric_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate test set based on different metrics")

    parser.add_argument("dataset_path", help="Path to dataset", type=Path)

    parser.add_argument("--output_dir",
                        help="Path where processing output will be stored",
                        default="out",
                        type=Path)

    parser.add_argument("--compute_metrics",
                        help="Toggle computation of metrics during evaluation",
                        default=True,
                        type=bool)                        

    parser.add_argument("--filter_by",
                        help="Enter list of patients to filter by",
                        nargs='+')

    parser.add_argument("--cores", help="Number of cores for multiprocessing", default=1, type=int)

    parser.add_argument("-v",
                        "--verbose",
                        dest="loglevel",
                        help="set loglevel to INFO",
                        action="store_const",
                        const=logging.INFO)

    parser.add_argument("-vv",
                        "--very-verbose",
                        dest="loglevel",
                        help="set loglevel to DEBUG",
                        action="store_const",
                        const=logging.DEBUG)

    args = parser.parse_args()
    setup_logging(args.loglevel)
    main(args)