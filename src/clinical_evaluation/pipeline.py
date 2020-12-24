import SimpleITK as sitk
from pathlib import Path
import traceback
import logging
from pathlib import Path
from typing import Union

from clinical_evaluation import preprocess
from clinical_evaluation import registration_methods

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

        except BaseException as e:
            _logger.error(f'Failed to load image using SimpleITK with error: \n {traceback.print_exc()}')
            raise e

        return image


    def deform(self, source: sitk.Image, target: sitk.Image, params: Union[Path, str], mode: str = 'ITKv4'):
        
        registration = REGISTRATION_MAP[mode]

        

        try:
            if params is not None:
                result = registration(source, target, params)
            else:
                result = registration(source, target)
        except ImportError as e:
            _logger.warning('This is caused most likely due to not having the SimpleElastix build of SimpleITK. \n \
                            Check the releases page to obtain a bdist for the SimpleITK build for Ubuntu \n \
                        Falling back to SimpleITK affine registration')

            result = REGISTRATION_MAP['ITKv4'](source, target)

        return result


    def save(self, image: sitk.Image, output_dir: Path, tag: str= 'deformed'):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = (output_dir / tag ).with_suffix(".nrrd")
        sitk.WriteImage(image, str(save_path), True)
        _logger.info(f"Saved deformed image to : {str(save_path)}")


    def preprocess(self, image: sitk.Image, preprocess_fn=None):
        if preprocess_fn is not None:
            fn = getattr(preprocess, preprocess_fn)
            return fn(image)
            
    def apply_body_mask(self, image: sitk.Image, HU_threshold: int = -300):
        array = sitk.GetArrayFromImage(image)
        array = preprocess.apply_body_mask_and_bound(array, masking_value=-1024, \
                                apply_mask=True, apply_bound=False, HU_threshold=HU_threshold)

        masked_image = sitk.GetImageFromArray(array)
        masked_image.CopyInformation(image)

        return masked_image