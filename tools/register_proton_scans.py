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
from clinical_evaluation.registration_tools import (RegistrationInformation, metrics, pipeline,
                                                    preprocess, regviz, utils)
from clinical_evaluation.utils.logging import setup_logging
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
    reginfo_data = RegistrationInformation(outdir=out_folder)
    eval_pipeline = pipeline.EvaluationPipeline()

    for patient in tqdm(mapping_json):
        patient_scans = mapping_json[patient]
        folder = (valid_folder / patient).resolve()
        assert folder.exists(), "Patient: {patient} does not exist in the directory provided."

        # For each pair among a single patient
        for idx, pair in tqdm(enumerate(patient_scans)):
            CT_path = folder / "CT" / pair["rCT"][1] / "CT.nrrd"
            mask_paths = {path.stem: path for path in (folder / "CT" / pair["rCT"][1]).glob("*.nrrd") \
            CBCT_path = folder / "CBCT" / pair["CBCT"][1] / "CBCT.nrrd"

            # Load CT, clip values
            CT = eval_pipeline.load(CT_path)
            CT = preprocess.clip_values(CT)

            # Loading RT masks that are present with planning CT
            rt_masks = {}
            for label, path in mask_paths.items():
                rt_masks[label] = eval_pipeline.load(path)
                rt_masks[label].CopyInformation(CT)

            # If a BODY mask exists in the RT masks, use it, 
            # otherwise generate a body mask 
            if "BODY" in rt_masks:
                CT_mask = rt_masks["BODY"]
            else:
                CT_mask = eval_pipeline.get_body_mask(CT)
                
            # Load CBCT, clip and generate body mask
            CBCT = eval_pipeline.load(CBCT_path)
            CBCT = preprocess.clip_values(CBCT)
            CBCT_mask = eval_pipeline.get_body_mask(CBCT, HU_threshold=-700)

            # Perform deformable registration using SimpleElastix parameters:
            # https://elastix.lumc.nl/modelzoo/par0032/
            
            # In addition to scans, masks for patient body in both CT 
            # and CBCT scans is provided. 
            params = {
            "config": ["/home/suraj/Repositories/clinical-evaluation/elastix_params/Par0032_rigid.txt", \
                        "/home/suraj/Repositories/clinical-evaluation/elastix_params/Par0032_bsplines.txt"],
            "target_mask": CBCT_mask,
            "source_mask": CT_mask
            }

            logger.info(f"Registering scans: {CBCT_path} and {CT_path}")

             # Deform the CT to CBCT and obtained the deformed planning CT or dpCT
            dpCT, elastixfilter = eval_pipeline.deform(CT, CBCT, params, mode='Elastix')

            # Propagate CBCT mask to dpCT for better correspondence
            dpCT = utils.apply_mask(dpCT, CBCT_mask)

            # Apply body masks to CBCT and CT
            CBCT = utils.apply_mask(CBCT, CBCT_mask)
            CT = utils.apply_mask(CT, CT_mask)

            # Deform masks/ propagate contours based on the deformation fields.
            rt_masks = {k: sitk.Cast(sitk.Transformix(mask, elastixfilter.GetTransformParameterMap()), sitk.sitkInt8)  \
                                                                            for k, mask in rt_masks.items()}

            # Save all the required image
            logger.info("Complete! Saving output")
            out_dir = out_folder / folder.stem / str(idx)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save CBCT, CT and deformed CT
            sitk.WriteImage(CBCT, str(out_dir / f"target.nrrd"), True)
            sitk.WriteImage(CT, str(out_dir / f"source.nrrd"), True)
            sitk.WriteImage(dpCT, str(out_dir / f"deformed.nrrd"), True)

            # Save all the propagated contours
            for label, mask in rt_masks.items():
                sitk.WriteImage(mask, str(out_dir / f"{label}.nrrd"), True)

            if args.analyze:
                # Calculate metrics between CBCT (target) and dpCT (deformed)
                metric_dict = metrics.calculate_metrics(CBCT, dpCT, offset=-1000)

                for label, mask in rt_masks.items():
                    mask_metrics = metrics.calculate_metrics(CBCT, dpCT, mask=mask, offset=-1000)
                    metric_dict.update({f"{k}_{label}": v for k, v in mask_metrics.items()})

                metric_dict["save_dir"] = str(out_dir)
                metric_dict["source"] = pair
                metric_dict["Patient"] = folder.stem
                metric_dict["Pair#"] = idx
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
    parser.add_argument("mapping_json",
                        help="Path to json file containing pair mappings for CBCT-CT",
                        type=Path)

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

    setup_logging(args.loglevel)
    main(args)
