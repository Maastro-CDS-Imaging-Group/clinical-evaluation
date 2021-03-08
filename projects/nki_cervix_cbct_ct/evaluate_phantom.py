from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from clinical_evaluation.evaluation_tools import phantom_evaluator
from clinical_evaluation.utils import file_utils


def main(args):
    phantoms_data_path = args.phantoms_data_path.resolve()
    translated_data_path = args.translated_data_path.resolve()

    phantom_values = {
            "Air": -1000,
            "LDPE": -100,
            "Polystyrene": -35,
            "Acrylic": 120,
            "Delrin": 340,
            "Teflon": 950,
            "plate": 0
        }

    phantom_values = {k:v + 1024 for k,v in phantom_values.items()}


    for phantom_folder in phantoms_data_path.iterdir():
        assert phantom_folder.stem in [folder.stem for folder in translated_data_path.iterdir()], \
            f"translated folder for {phantom_folder} not found"
        
        original_path = phantom_folder / "CT.nrrd"
        original = sitk.ReadImage(str(original_path.resolve()))

        translated_path = translated_data_path / phantom_folder.stem / "CT.nrrd"
        translated = sitk.ReadImage(str(translated_path.resolve()))

        # Shift by 1024 to get all positive values
        original = original + 1024
        translated = translated + 1024

        insert_masks = {}
        for path in phantom_folder.glob("*.nrrd"):
            if path.stem != "CT":
                insert_masks[path.stem] = {
                    "image": sitk.ReadImage(str(path.resolve())),
                    "value": phantom_values[path.stem]
                }

        evaluate_phantom = phantom_evaluator.PhantomEvaluator(original, translated,\
                                 inserts_dict=insert_masks)

        metric_dict = evaluate_phantom()
        df = pd.json_normalize(metric_dict)
        df = file_utils.convert_to_multilevel_df(df)
        df = df.transpose()
        df.to_csv(f'{phantom_folder.stem}_evaluation.csv')





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("phantoms_data_path", help="Path to original dataset", type=Path)
    parser.add_argument("translated_data_path", help="Path to translated dataset", type=Path)

    args = parser.parse_args()

    main(args)





