"""
Register, Visualize and Analyze Registrations. 

Register using SimpleElastix or ITKv4
Visualize using SimpleITK Checkboard, Overlay with image and video outputs
Analyze using metrics for HU differences, SSIM, PSNR, MSE, RMSE

"""
import logging
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.evaluation import metrics
from clinical_evaluation.registration import CSVSaver, pipeline
from clinical_evaluation.utils import ops, preprocess, visualizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(args):
    in_folder = args.dataset_path.resolve()
    out_folder = args.output_dir.resolve()

    reginfo_data = CSVSaver(outdir=out_folder)
    reg_pipeline = pipeline.RegistrationPipeline()

    # Each dir is a patient with a single CBCT-CT pair
    for patient in tqdm([x for x in in_folder.iterdir() if x.is_dir()]):
        # There's only one CT and CBCT per patient
        CT_path = list(patient.rglob("CT.nrrd"))[0]
        CBCT_path = list(patient.rglob("CBCT.nrrd"))[0]

        # Load CT, clip values
        CT = reg_pipeline.load(CT_path)
        CT = preprocess.clip_values(CT)
        # Generate a body mask 
        CT_mask = reg_pipeline.get_body_mask(CT)

        # Load CBCT, clip and generate body mask
        CBCT = reg_pipeline.load(CBCT_path)
        CBCT = preprocess.clip_values(CBCT)
        CBCT_mask = reg_pipeline.get_body_mask(CBCT, HU_threshold=-700)

        # Apply body masks to CBCT and CT
        CBCT = ops.apply_mask(CBCT, CBCT_mask)
        CT = ops.apply_mask(CT, CT_mask)

        # Perform deformable registration using SimpleElastix parameters:
        # https://elastix.lumc.nl/modelzoo/par0032/

        params = {
            "config": [
                "./elastix_params/Par0032_rigid.txt",
                "./elastix_params/Par0032_bsplines.txt"
            ]
        
        }

        logger.info(f"Registering scans: {CBCT_path} and {CT_path}")

            # Deform the CT to CBCT and obtained the deformed planning CT or dpCT
        dpCT, elastixfilter = reg_pipeline.deform(CT, CBCT, params, mode='Elastix')

        # Propagate CBCT mask to dpCT for better correspondence
        dpCT = ops.apply_mask(dpCT, CBCT_mask)


        # Save all the required image
        out_dir = out_folder / patient.name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Complete! Saving output to {str(out_dir)}")


        # Save CBCT, CT and deformed CT
        sitk.WriteImage(CBCT, str(out_dir / f"target.nrrd"), True)
        sitk.WriteImage(CT, str(out_dir / f"source.nrrd"), True)
        sitk.WriteImage(dpCT, str(out_dir / f"deformed.nrrd"), True)

        if args.analyze:
            # Calculate metrics between CBCT (target) and dpCT (deformed)
            metric_dict = metrics.calculate_metrics(CBCT, dpCT)

            metric_dict["save_dir"] = str(out_dir)
            metric_dict["Patient"] = patient.name
            reginfo_data.add_info(metric_dict)

        if args.visualize:
            # Generate and save registration visualizations
            viz = visualizer.Visualizer(outdir=out_dir, save_mode='image+video')
            viz.save_visualizations(CBCT, dpCT, checkerboard=True, overlay=True)

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

    parser.add_argument("-z", "--visualize",
                        help="If registration process should be visualized",
                        action='store_true', default=False)

                        
    parser.add_argument("-a", "--analyze",
                        help="If registration process should be analyzed",
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
