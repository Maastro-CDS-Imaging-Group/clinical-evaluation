"""
Register, Visualize and Analyze Registrations. 

Register using SimpleElastix or ITKv4
Visualize using SimpleITK Checkboard, Overlay with image and video outputs
Analyze using metrics for HU differences, SSIM, PSNR, MSE, RMSE

"""
import json
import logging
import sys
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.evaluation import metrics
from clinical_evaluation.registration import CSVSaver, pipeline
from clinical_evaluation.utils import ops, preprocess, visualizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

def main(args):
    valid_folder = args.dataset_path.resolve()

    # Load json file showing mappings between rCT and 
    # CBCTs
    mapping_json_path = args.mapping_json.resolve()
    with open(mapping_json_path) as fp:
        mapping_json = json.load(fp)

    out_folder = args.output_dir.resolve()
    reginfo_data = CSVSaver(outdir=out_folder)
    reg_pipeline = pipeline.RegistrationPipeline()

    for patient in tqdm(mapping_json):
        patient_scans = mapping_json[patient]
        folder = (valid_folder / patient).resolve()
        assert folder.exists(), "Patient: {patient} does not exist in the directory provided."

        # For each pair among a single patient
        for idx, pair in tqdm(enumerate(patient_scans)):
            CT_path = folder / "CT" / pair["rCT"][1] / "CT.nrrd"
            mask_paths = {path.stem: path for path in (folder / "CT" / pair["rCT"][1]).glob("*.nrrd")}
            CBCT_path = folder / "CBCT" / pair["CBCT"][1] / "CBCT.nrrd"

            # Load CT, clip values
            CT = reg_pipeline.load(CT_path)
            CT = preprocess.clip_values(CT)

            # Loading RT masks that are present with planning CT
            rt_masks = {}
            for label, path in mask_paths.items():
                rt_masks[label] = reg_pipeline.load(path)
                rt_masks[label].CopyInformation(CT)

            # If a BODY mask exists in the RT masks, use it, 
            # otherwise generate a body mask 
            if "BODY" in rt_masks:
                CT_mask = rt_masks["BODY"]
            else:
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
            "config": ["./configs/Par0032_rigid.txt", \
                        "./configs/Par0032_bsplines.txt"]
            }

            logger.info(f"Registering scans: {CBCT_path} and {CT_path}")

             # Deform the CT to CBCT and obtained the deformed planning CT or dpCT
            dpCT, elastixfilter = reg_pipeline.deform(CT, CBCT, params, mode='Elastix')

            # Propagate CBCT mask to dpCT for better correspondence
            dpCT = ops.apply_mask(dpCT, CBCT_mask)


            # Save all the required image
            logger.info("Complete! Saving output")
            out_dir = out_folder / folder.stem / str(idx)
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.propagate_contours:
                logger.info("Propagating contours ...")
                # Deform masks/ propagate contours based on the deformation fields.
                rt_masks = {k: sitk.Cast(sitk.Transformix(mask, elastixfilter.GetTransformParameterMap()), sitk.sitkInt8)  \
                                                                                for k, mask in rt_masks.items()}
            else:
                rt_masks = {}
     

            # Save CBCT, CT and deformed CT
            sitk.WriteImage(CBCT, str(out_dir / f"target.nrrd"), True)
            sitk.WriteImage(CT, str(out_dir / f"source.nrrd"), True)
            sitk.WriteImage(dpCT, str(out_dir / f"deformed.nrrd"), True)

            # Save all the propagated contours
            for label, mask in rt_masks.items():
                sitk.WriteImage(mask, str(out_dir / f"{label}.nrrd"), True)

            if args.analyze:
                # Calculate metrics between CBCT (target) and dpCT (deformed)
                metric_dict = metrics.calculate_metrics(CBCT, dpCT)

                for label, mask in rt_masks.items():
                    mask_metrics = metrics.calculate_metrics(CBCT, dpCT, mask=mask)
                    metric_dict.update({f"{k}_{label}": v for k, v in mask_metrics.items()})

                metric_dict["save_dir"] = str(out_dir)
                metric_dict["source"] = pair
                metric_dict["Patient"] = folder.stem
                metric_dict["Pair#"] = idx
                reginfo_data.add_info(metric_dict)

            if args.visualize:
                # Generate and save registration visualizations
                viz = visualizer.Visualizer(outdir=out_dir, save_mode='image+video')
                viz.save_visualizations(CBCT, dpCT, checkerboard=True, overlay=True)

            reginfo_data.save_info()


if __name__ == "__main__":
    from clinical_evaluation.utils.logging import setup_logging

    import argparse
    parser = argparse.ArgumentParser(
        description="Registration + Visualization + Analysis for scans in a CBCT-CT dataset")

    parser.add_argument("dataset_path", help="Path to dataset", type=Path)

    parser.add_argument("--mapping_json",
                        help="Path to mapping json file",
                        type=Path)

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


    parser.add_argument("--propagate_contours",
                        help="If contours should also be propagated from planning CT",
                        action='store_false', default=False)

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
