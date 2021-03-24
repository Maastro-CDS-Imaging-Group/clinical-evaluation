"""
Register, Visualize and Analyze Registrations. 

Register using SimpleElastix or ITKv4
Visualize using SimpleITK Checkboard, Overlay with image and video outputs
Analyze using metrics for HU differences, SSIM, PSNR, MSE, RMSE

"""
import logging
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.registration_tools import (pipeline, regviz)
from clinical_evaluation.utils import (metrics, preprocess, ops)
from clinical_evaluation.registration_tools import RegistrationInformation

from tqdm import tqdm
from clinical_evaluation.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main(args):
    data_folder = args.dataset_path.resolve()
    reginfo_data = RegistrationInformation(outdir=data_folder)

    for folder in tqdm(list(data_folder.iterdir())):
        translated = folder / "translated.nrrd"
        deformed = folder / "deformed.nrrd"
        original = folder / "target.nrrd"

        if not (translated.exists() and deformed.exists() and original.exists()):
            logger.warning(f"Skipping {folder} as all required nrrd files are not present")
            continue

        CT = sitk.ReadImage(str(deformed))
        CBCT = sitk.ReadImage(str(original))
        CT = sitk.ReadImage(str(translated))

        metric_dict = {}

        original_metrics = metrics.calculate_metrics(CT, CBCT)
        translated_metrics = metrics.calculate_metrics(CT, sCT)

        metric_dict.update({f"original_{k}": v for k, v in original_metrics.items()})
        metric_dict.update({f"translated_{k}": v for k, v in translated_metrics.items()})

        if args.enable_masks:

            rt_masks = {}
            for fn in folder.glob("*.nrrd"):
                if fn.stem not in ["target", "translated", "deformed", "source"]:
                    mask_image = sitk.ReadImage(str(fn))
                    rt_masks[fn.stem] = mask_image

            for label, mask in rt_masks.items():

                original_mask_metrics = metrics.calculate_metrics(CT, CBCT, mask=mask)
                metric_dict.update({f"original_{k}_{label}": v for k,v in original_mask_metrics.items()})

                translated_mask_metrics = metrics.calculate_metrics(CT, sCT, mask=mask)
                metric_dict.update({f"translated_{k}_{label}": v for k,v in translated_mask_metrics.items()})

        metric_dict["save_dir"] = str(data_folder)
        metric_dict["Patient"] = folder.stem
        reginfo_data.add_info(metric_dict)
        reginfo_data.save_info()
    
    mean_df = reginfo_data.get_aggregate_dataframe()
    mean_df = mean_df.transpose()

    print(mean_df)
    mean_df.to_csv(data_folder / "mean_test_metrics.csv")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Registration + Visualization + Analysis for scans in a CBCT-CT dataset")

    parser.add_argument("dataset_path", help="Path to dataset", type=Path)


    parser.add_argument("--enable_masks",
                        help="If comparisons should also be done for masks",
                        action='store_true', default=False)                        

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