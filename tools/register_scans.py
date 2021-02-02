import logging
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.registration_tools import (metrics, pipeline,
                                                    preprocess, regviz, utils)
from tqdm import tqdm

logger = logging.getLogger(__name__)

def main(args):
    eval_pipeline = pipeline.EvaluationPipeline()
    valid_folder = args.dataset_path.resolve()
    out_folder = args.output_dir.resolve()

    for folder in tqdm(list(valid_folder.iterdir())):
        CT_path = list((folder / "CT").rglob("CT.nrrd"))[0]
        CT = eval_pipeline.load(CT_path)
        CT, _ = eval_pipeline.apply_body_mask(CT)
            
        CBCT_path = (folder / "CBCT" / "X01").with_suffix(".nrrd")
        
        try:
            CBCT = eval_pipeline.load(CBCT_path)

        except:
            logger.error(f"Skipping: {folder.stem}")
            continue            

        CBCT = preprocess.hu_correction(CBCT)
        CBCT, mask = eval_pipeline.apply_body_mask(CBCT, HU_threshold=-700)

        # Perform deformable registration
        params = {
        "config": ["/home/suraj/Repositories/clinical-evaluation/elastix_params/Par0032_rigid.txt", \
                        "/home/suraj/Repositories/clinical-evaluation/elastix_params/Par0032_bsplines.txt"],
        }

        logger.info(f"Registering scans: {CBCT_path} and {CT_path}")
        dpCT, elastixImageFilter = eval_pipeline.deform(CT, CBCT, params, mode='Elastix')

        # Propagate CBCT mask to dpCT for better alignment
        dpCT = utils.apply_mask(dpCT, mask)
        
        # Save all the required image
        logger.info(f"Complete! Saving output")
        out_dir = out_folder / folder.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(CBCT, str(out_dir / "source.nrrd"), True)
        sitk.WriteImage(CT, str(out_dir / "target.nrrd"), True)
        sitk.WriteImage(dpCT, str(out_dir / "deformed.nrrd"), True)

        # Generate and save registration visualizations
        visualizer = regviz.RegistrationVisualizer(outdir=out_dir, save_mode='image+video')
        visualizer.save_registration_visualizations(CBCT, dpCT)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=
        "Register scans in a CBCT-CT dataset")

    parser.add_argument("dataset_path", help="Path to dataset", type=Path)
    parser.add_argument("--output_dir", help="Path where processing output will be stored", default="registered_scans", type=Path)
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

