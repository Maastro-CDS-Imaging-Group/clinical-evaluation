import logging
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.evaluation import metrics
from clinical_evaluation.registration import pipeline
from clinical_evaluation.utils import ops, preprocess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def save_visuals(visuals, save_dir, prefix=""):
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = str(save_dir / prefix)
    for key, visual in visuals.items():
        f = plt.figure(figsize=(15, 15))

        if "difference" in key:
            plt.imshow(visual,  cmap='coolwarm', vmin=-350, vmax=350)
            cb = plt.colorbar(orientation="horizontal",anchor=(1.0,0.0), ticks=[-350, 0, 350], fraction=0.03)
            cb.ax.tick_params(labelsize=30)

        else:
            plt.imshow(visual, cmap='gray')

        plt.axis('off')

        plt.savefig(f"{save_path}_{key}.png", pad_inches=0, bbox_inches='tight')


def main(args):
    data_folder = args.data_folder.resolve()
    save_dir = args.save_dir if args.save_dir else data_folder

    model_folders = list(x for x in data_folder.glob('*/') if x.is_dir())
 
    for model_folder in model_folders:
        patient_folders = list(x for x in model_folder.glob('*/') if x.is_dir())

        for idx, patient in enumerate(patient_folders):
            print(patient)
            translated = patient / "translated.nrrd"
            deformed = patient / "deformed.nrrd"
            
            save_to = save_dir / model_folder.stem / patient.stem
            CT = sitk.ReadImage(str(deformed))
            CT = ops.make_isotropic(CT)
            CT = sitk.GetArrayFromImage(CT)

            
            if idx == 0:
                original = patient / "target.nrrd"
                CBCT = sitk.ReadImage(str(original))
                CBCT = sitk.ReadImage(str(original))
                CBCT = ops.make_isotropic(CBCT)
                CBCT = sitk.GetArrayFromImage(CBCT)
                visuals = ops.get_visuals(CT, CBCT)

                save_visuals(visuals, save_to, prefix="Original")
                
                visuals = ops.get_visuals(CT, CT)

                save_visuals(visuals, save_to, prefix="Target")


            sCT = sitk.ReadImage(str(translated))
            sCT = ops.make_isotropic(sCT)
            sCT = sitk.GetArrayFromImage(sCT)

            visuals = ops.get_visuals(CT, sCT)
            save_visuals(visuals, save_to, prefix="Translated")


if __name__ == "__main__":
    from clinical_evaluation.utils.logging import setup_logging

    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Visualizations for original and predicted")

    parser.add_argument("data_folder", help="Path to dataset", type=Path)
    parser.add_argument("--save_dir",
                        help="Path where processing output will be stored",
                        type=Path)
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
