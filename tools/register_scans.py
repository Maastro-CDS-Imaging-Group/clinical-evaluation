"""
Register, Visualize and Analyze Registrations. 

Register using SimpleElastix or ITKv4
Visualize using SimpleITK Checkboard, Overlay with image and video outputs
Analyze using metrics for HU differences, SSIM, PSNR, MSE, RMSE

"""
import logging
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.registration_tools import (metrics, pipeline, preprocess, regviz, utils)
from clinical_evaluation.registration_tools import RegistrationInformation

from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(args):
    valid_folder = args.dataset_path.resolve()
    out_folder = args.output_dir.resolve()
    reginfo_data = RegistrationInformation(outdir=out_folder)

    eval_pipeline = pipeline.EvaluationPipeline()

    for folder in tqdm(list(valid_folder.iterdir())):
        # Get first match with filename CT.nrrd
        CT_path = list((folder / "CT").rglob("CT.nrrd"))[0]
        CT = eval_pipeline.load(CT_path)
        CT, _ = eval_pipeline.apply_body_mask(CT)

        # Get CBCT path and try to load the CBCT if it exists
        CBCT_path = (folder / "CBCT" / "X01").with_suffix(".nrrd")
        try:
            CBCT = eval_pipeline.load(CBCT_path)
        except RuntimeError:
            logger.error(f"Skipping: {folder.stem}")
            continue

        # Correct the CBCT value if not calibrated
        CBCT = preprocess.hu_correction(CBCT, cval=-1024)
        CBCT, mask = eval_pipeline.apply_body_mask(CBCT, HU_threshold=-700)

        # Perform deformable registration using SimpleElastix parameters:
        # https://elastix.lumc.nl/modelzoo/par0032/
        params = {
        "config": ["/home/suraj/Repositories/clinical-evaluation/elastix_params/Par0032_rigid.txt", \
                        "/home/suraj/Repositories/clinical-evaluation/elastix_params/Par0032_bsplines.txt"],
        }

        logger.info(f"Registering scans: {CBCT_path} and {CT_path}")

        # Deform the CT to CBCT
        dpCT, _ = eval_pipeline.deform(CT, CBCT, params, mode='Elastix')

        # Propagate CBCT mask to dpCT for better alignment
        dpCT = utils.apply_mask(dpCT, mask)

        # Save all the required image
        logger.info("Complete! Saving output")
        out_dir = out_folder / folder.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save CBCT, CT and deformed CT
        sitk.WriteImage(CBCT, str(out_dir / "source.nrrd"), True)
        sitk.WriteImage(CT, str(out_dir / "target.nrrd"), True)
        sitk.WriteImage(dpCT, str(out_dir / "deformed.nrrd"), True)

        if args.analyze:
            # Calculate metrics between CBCT (target) and dpCT (deformed)
            metric_dict = metrics.calculate_metrics(CBCT, dpCT)
            metric_dict["save_dir"] = str(out_dir)
            metric_dict["Patient"] = folder.stem
            
            reginfo_data.add_info(metric_dict)

        if args.visualize:
            # Generate and save registration visualizations
            visualizer = regviz.RegistrationVisualizer(outdir=out_dir, save_mode='image+video')
            visualizer.save_registration_visualizations(CBCT, dpCT)

        reginfo_data.save_info()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Registration + Visualization + Analysis for scans in a CBCT-CT dataset")

    parser.add_argument("dataset_path", help="Path to dataset", type=Path)
    parser.add_argument("--output_dir",
                        help="Path where processing output will be stored",
                        default="registered_scans",
                        type=Path)

    parser.add_argument("--visualize",
                        help="If registration process should be visualized",
                        default=True,
                        type=bool)
    parser.add_argument("--analyze",
                        help="If registration process should be analyzed",
                        default=True,
                        type=bool)

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

    main(args)
