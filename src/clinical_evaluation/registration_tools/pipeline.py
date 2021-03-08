import logging
import traceback
from pathlib import Path
from typing import Union

import SimpleITK as sitk
from clinical_evaluation.utils import (preprocess, ops)
from clinical_evaluation.registration_tools import registration_methods

_logger = logging.getLogger(__name__)

REGISTRATION_MAP = {
    "ITKv4": registration_methods.registration_ITKv4,
    "Elastix": registration_methods.registration_Elastix
}


class EvaluationPipeline:
    def load(self, path: Union[Path, str]):
        _logger.info(f"Loading image from {path}")
        try:
            image = sitk.ReadImage(str(path))

        except:
            _logger.debug(
                f'Failed to load image using SimpleITK with error: \n {traceback.print_exc()}')
            raise RuntimeError('Failed to load image using SimpleITK')

        return image

    def deform(self,
               source: sitk.Image,
               target: sitk.Image,
               params: Union[dict, str],
               mode: str = 'ITKv4'):

        registration = REGISTRATION_MAP[mode]

        try:
            if params is not None:
                result = registration(source, target, params)
            else:
                result = registration(source, target)
        except ImportError:
            _logger.warning(
                'This is caused most likely due to not having the SimpleElastix build of SimpleITK. \n \
                            Check the releases page to obtain a bdist for the SimpleITK build for Ubuntu \n \
                        Falling back to SimpleITK affine registration')

            result = REGISTRATION_MAP['ITKv4'](source, target)

        return result

    def save(self, image: sitk.Image, output_dir: Path, tag: str = 'deformed'):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = (output_dir / tag).with_suffix(".nrrd")
        sitk.WriteImage(image, str(save_path), True)
        _logger.info(f"Saved deformed image to : {str(save_path)}")

    def preprocess(self, image: sitk.Image, preprocess_fn=None):
        if preprocess_fn is not None:
            fn = getattr(preprocess, preprocess_fn)
            return fn(image)

    def get_body_mask(self, image: sitk.Image, HU_threshold: int = -300):
        array = sitk.GetArrayFromImage(image)
        mask = preprocess.get_body_mask(array, \
                                HU_threshold=HU_threshold)

        mask = sitk.GetImageFromArray(mask)
        mask.CopyInformation(image)

        return mask
