import logging
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.evaluation import metrics
from clinical_evaluation.registration import pipeline
from clinical_evaluation.utils import ops, preprocess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MIN_HU = -135
MAX_HU = 215

def save_visuals(visuals, save_dir, prefix=""):
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = str(save_dir / prefix)
    for key, visual in visuals.items():
        f = plt.figure(figsize=(15, 15))
        plt.imshow(visual, cmap='gray')
        plt.axis('off')
        plt.savefig(f"{save_path}_{key}.png", pad_inches=0, bbox_inches='tight')


def main(args):
    data_folder = args.data_folder.resolve()
    save_dir = args.save_dir if args.save_dir else data_folder

    model_folders = list(x for x in data_folder.glob('*/') if x.is_dir())
 
    for idx, model_folder in enumerate(model_folders):
        patient_folders = list(x for x in model_folder.glob('*/') if x.is_dir())

        for patient in patient_folders:
            translated = patient / "translated.nrrd"
            deformed = patient / "deformed.nrrd"
            
            save_to = save_dir / patient.stem
            window_save_to = save_dir / patient.stem / "windowed"
            CT = sitk.ReadImage(str(deformed))

            if args.bound_mask:
                mask = preprocess.get_body_mask(CT, -300)
                bounds = ops.get_bounds_from_mask(mask)
            else:
                bounds = None
            
            if bounds:
                CT = ops.slice_image(CT, bounds[0], bounds[1])   
            CT = ops.make_isotropic(CT)

            if idx == 0:
                original = patient / "target.nrrd"
                CBCT = sitk.ReadImage(str(original))
                if bounds:
                    CBCT = ops.slice_image(CBCT, bounds[0], bounds[1])                
                CBCT = ops.make_isotropic(CBCT)

                
       
                visuals = {f"image": ops.get_image_preview(CBCT, orientation='horizontal')}
                save_visuals(visuals, save_to, prefix="Original")
                visuals = {f"windowed_image": ops.get_image_preview(CBCT, min_HU=MIN_HU, max_HU=MAX_HU, orientation='horizontal')}
                save_visuals(visuals, window_save_to, prefix="Original")
            
                visuals = {f"image": ops.get_image_preview(CT, orientation='horizontal')}
                save_visuals(visuals, save_to, prefix="Target")
                visuals = {f"windowed_image": ops.get_image_preview(CT, min_HU=MIN_HU, max_HU=MAX_HU, orientation='horizontal')}
                save_visuals(visuals, window_save_to, prefix="Target")


            sCT = sitk.ReadImage(str(translated))
            if bounds:
                sCT = ops.slice_image(sCT, bounds[0], bounds[1])                
            sCT = ops.make_isotropic(sCT)

            visuals = {f"{model_folder.stem}": ops.get_image_preview(sCT, orientation='horizontal')}
            save_visuals(visuals, save_to, prefix="Translated")
            visuals = {f"windowed_{model_folder.stem}": ops.get_image_preview(sCT, min_HU=MIN_HU, max_HU=MAX_HU, orientation='horizontal')}
            save_visuals(visuals, window_save_to, prefix="Translated")
            

if __name__ == "__main__":
    from clinical_evaluation.utils.logging import setup_logging

    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Visualizations for original and predicted")

    parser.add_argument("data_folder", help="Path to dataset", type=Path)
    parser.add_argument("--save_dir",
                        help="Path where processing output will be stored",
                        type=Path)

    parser.add_argument("--bound_mask",
                        help="If the image should be bounded based on body mask",
                        default=False)         

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
