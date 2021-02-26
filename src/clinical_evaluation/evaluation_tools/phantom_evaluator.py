import SimpleITK as sitk
from pathlib import Path
import numpy as np
from clinical_evaluation.utils import metrics, preprocess, ops


class PhantomEvaluator:
    def __init__(self, original_path: Path, translated_path: Path, \
                    target_phantom_path: Path = None, \
                    inserts_dict: dict = None):

        assert (inserts_dict is not None or target_phantom_path is not None), \
            "Either insert masks or target phantom needs to be provided for evaluation"
        
        self.original = sitk.ReadImage(str(original_path.resolve()))
        self.translated = sitk.ReadImage(str(translated_path.resolve()))

        self.insert_masks = {}
        self.target_phantom = None

        if inserts_dict is not None:
            assert isinstance(inserts_dict, dict), \
                "A dict of the format {'insert': {'path': '...', 'value': 300}, ..} " \
                "must be provided"

            for label, insert_dict in inserts_dict.items():
                assert all(k in insert_dict for k in ['path', 'value'])
                self.insert_masks[label] = {
                    'mask': sitk.ReadImage(str(insert_dict['path'])),
                    'value': insert_dict['value']
                }

        if target_phantom_path is not None:
            self.target_phantom = sitk.ReadImage(str(translated_path.resolve()))

    def apply_phantom_mask(self, image: sitk.Image, HU_threshold: int = -300):
        array = sitk.GetArrayFromImage(image)
        mask = preprocess.get_body_mask(array, \
                                HU_threshold=HU_threshold)
        mask = sitk.GetImageFromArray(mask)
        mask.CopyInformation(image)

        return ops.apply_mask(image, mask)


    def __call__(self):
        self.metric_dict = {}

        self.original = self.apply_phantom_mask(self.original)
        self.translated = self.apply_phantom_mask(self.translated)

        # If insert masks are provided compute masked metrics

        if self.insert_masks:
            original = sitk.GetArrayFromImage(self.original)
            translated = sitk.GetArrayFromImage(self.translated)                                                

            for label, insert_dict in self.insert_masks.items():
                mask = sitk.GetArrayFromImage(insert_dict['mask'])
                mask = ~mask.astype(np.bool)

                original_value = np.ma.masked_array(original, mask=mask)
                translated_value = np.ma.masked_array(translated, mask=mask)

                self.metric_dict[label] = {
                    'original': metrics.mae(insert_dict['value'], original_value),
                    'translated': metrics.mae(insert_dict['value'], translated_value)
                }

            # Compute averaged values over all inserts + plate in the phantom
            self.metric_dict['Overall'] = {
                'original': np.mean([v['original'] for k, v in self.metric_dict.items()]),
                'translated': np.mean([v['translated'] for k, v in self.metric_dict.items()])

            }

        if self.target_phantom:
            self.target_phantom = self.apply_phantom_mask(self.target_phantom)
            target = sitk.GetArrayFromImage(self.target_phantom)

            self.metric_dict['Overall'] = {
                 'original': metrics.mae(target, original),
                'translated': metrics.mae(target, translated)
            }

        return self.metric_dict
        