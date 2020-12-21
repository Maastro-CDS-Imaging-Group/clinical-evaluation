import SimpleITK as sitk
from pathlib import Path
import traceback
import logging
from pathlib import Path
from typing import Union
from functools import partial

import preprocess

_logger = logging.getLogger(__name__)

class EvaluationPipeline:
    def load(self, path: Union[Path, str]):
        _logger.info(f"Loading image from {path}")
        try:
            image = sitk.ReadImage(str(path))

        except BaseException as e:
            _logger.error(f'Failed to load image using SimpleITK with error: \n {traceback.print_exc()}')
            raise e

        return image


    def deform(self, source: sitk.Image, target: sitk.Image, params: Path):
        try:
            elastixImageFilter = sitk.ElastixImageFilter()
        except e:
            _logger.error('This is caused most likely due to not having the SimpleElastix build of SimpleITK. \n \
                            Check the releases page to obtain a bdist for the SimpleITK build for Ubuntu')
            raise e

        if params is not None:
            pmap = elastixImageFilter.ReadParameterFile(str(params))

        # Set moving and fixed images
        elastixImageFilter.SetFixedImage(target)
        elastixImageFilter.SetMovingImage(source)

        if params is not None:
            elastixImageFilter.SetParameterMap(pmap)

        result = elastixImageFilter.Execute()

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
            
