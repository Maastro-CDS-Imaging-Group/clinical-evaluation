import SimpleITK as sitk
from pathlib import Path
import numpy as np
from clinical_evaluation.utils import metrics, preprocess, ops


class PhantomEvaluator:
    def __init__(self, original: sitk.Image, translated: sitk.Image, \
                    target_phantom: sitk.Image = None, \
                    inserts_dict: dict = None):

        assert (inserts_dict is not None or target_phantom is not None), \
            "Either insert masks or target phantom needs to be provided for evaluation"
        
        self.original = original
        self.translated = translated

        self.insert_masks = {}
        self.target_phantom = None

        if inserts_dict is not None:
            assert isinstance(inserts_dict, dict), \
                "A dict of the format {'insert': {'image': '...', 'value': 300}, ..} " \
                "must be provided"

            for label, insert_dict in inserts_dict.items():
                assert all(k in insert_dict for k in ['image', 'value'])
                self.insert_masks[label] = {
                    'mask': insert_dict['image'],
                    'value': insert_dict['value']
                }

        if target_phantom is not None:
            self.target_phantom = target_phantom

    def apply_phantom_mask(self, image: sitk.Image, HU_threshold: int = -300):
        array = sitk.GetArrayFromImage(image)
        mask = preprocess.get_body_mask(array, \
                                HU_threshold=HU_threshold)
        mask = sitk.GetImageFromArray(mask)
        mask.CopyInformation(image)

        return ops.apply_mask(image, mask)


    def __call__(self):
        self.metric_dict = {}

        # If insert masks are provided compute masked metrics
        if self.insert_masks:
            original = sitk.GetArrayFromImage(self.original)
            translated = sitk.GetArrayFromImage(self.translated)                                                

            for label, insert_dict in self.insert_masks.items():
                mask = sitk.GetArrayFromImage(insert_dict['mask'])
                mask = ~mask.astype(np.bool)

                original_value = np.ma.masked_array(original, mask=mask)
                original_value = np.mean(original_value)
                
                translated_value = np.ma.masked_array(translated, mask=mask)
                translated_value = np.mean(translated_value)

                print(f"{label}, Original: {original_value}, Translated: {translated_value}")

                self.metric_dict[label] = {
                    'ideal': insert_dict['value'],
                    'CBCT': original_value,
                    'sCT': translated_value,
                    'CBCT_relative_difference': ((original_value - insert_dict['value']) / insert_dict['value']) * 100,
                    'sCT_relative_difference': ((translated_value - insert_dict['value']) / insert_dict['value']) * 100
                }


        if self.target_phantom:
            target = sitk.GetArrayFromImage(self.target_phantom)

            self.metric_dict['Overall'] = {
                 'original': metrics.mae(target, original),
                'translated': metrics.mae(target, translated),
                'original_relative_difference': metrics.relative_difference(target, original),
                'translated_relative_difference': metrics.relative_difference(target, translated)

            }

        return self.metric_dict
        