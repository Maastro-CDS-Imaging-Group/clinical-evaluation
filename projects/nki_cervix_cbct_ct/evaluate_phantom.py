from pathlib import Path
from clinical_evaluation.evaluation_tools import phantom_evaluator
from clinical_evaluation.utils import file_utils
import pandas as pd
import itertools

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


    for phantom_folder in phantoms_data_path.iterdir():
        assert phantom_folder.stem in [folder.stem for folder in translated_data_path.iterdir()], \
            f"translated folder for {phantom_folder} not found"
        
        original_path = phantom_folder / "CT.nrrd"
        translated_path = translated_data_path / phantom_folder.stem / "CT.nrrd"

        insert_mask_paths = {}
        for path in phantom_folder.glob("*.nrrd"):
            if path.stem != "CT":
                insert_mask_paths[path.stem] = {
                    "path": path,
                    "value": phantom_values[path.stem]
                }

        evaluate_phantom = phantom_evaluator.PhantomEvaluator(original_path, translated_path,\
                                 inserts_dict=insert_mask_paths)

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





