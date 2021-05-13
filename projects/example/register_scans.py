import logging
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.registration import pipeline, regviz
from clinical_evaluation.utils import metrics, ops, preprocess
from clinical_evaluation.utils.logging import setup_logging


def main(args):
    dataset_dir = args.dataset_path.resolve()
    out_folder = args.out_dir.resolve()

    eval_pipeline = pipeline.EvaluationPipeline()


    for patient in dataset_dir.iterdir():
        if patient.is_dir():

            # Get first match of the extension for CT and MR. Change this
            # behaviour if desired
            CT_path = list((patient / "CT").glob(f"*{args.extension}"))[0]
            MR_path = list((patient / "MR").glob(f"*{args.extension}"))[0]
                
            CT = eval_pipeline.load(CT_path)
            MR = eval_pipeline.load(MR_path)

            # Rigid register the MR to CT and obtained the registration fields
            deformed, transform = eval_pipeline.deform(MR, CT, "Rigid", mode='ITKv4')

            print("Complete! Saving output")
            out_dir = out_folder / patient.stem
            out_dir.mkdir(parents=True, exist_ok=True)

            sitk.WriteImage(CT, str(out_dir / f"target.nrrd"), True)
            sitk.WriteImage(MR, str(out_dir / f"source.nrrd"), True)
            sitk.WriteImage(deformed, str(out_dir / f"deformed.nrrd"), True)


            if args.visualize:
                visualizer = regviz.RegistrationVisualizer(outdir=out_dir, save_mode='image+video')
                visualizer.save_registration_visualizations(CT, deformed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Registration + Visualization + Analysis for scans in a CBCT-CT dataset")

    parser.add_argument("dataset_path", help="Path to dataset", type=Path)
    parser.add_argument("--out_dir",
                        help="Path where processing output will be stored",
                        default="registered_scans",
                        type=Path)

    parser.add_argument("--extension",
                        help="File extensions to search for",
                        default="nrrd")

    parser.add_argument("-z", "--visualize",
                        help="If registration process should be visualized",
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
