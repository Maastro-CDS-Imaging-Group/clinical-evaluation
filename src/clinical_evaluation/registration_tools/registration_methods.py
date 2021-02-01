import logging
import SimpleITK as sitk
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

TRANSFORMATION_MAP = {
    "Affine": sitk.AffineTransform(3),
    "Rigid": sitk.Euler3DTransform()
}


def registration_ITKv4(source: sitk.Image,
                       target: sitk.Image,
                       params: str = "Affine"):
    _logger.info("Running ITKv4 Registration ...")
    registration_transform = get_registration_transform(target, source, \
                                    registration_type=TRANSFORMATION_MAP[params])

    result = sitk.Resample(source, target, registration_transform,
                           sitk.sitkLinear, -1024, source.GetPixelID())
    return result, registration_transform


def registration_Elastix(source: sitk.Image, target: sitk.Image, params: dict):
    _logger.info("Running Elastix Registration ...")

    elastixImageFilter = sitk.ElastixImageFilter()
    # Set moving and fixed images
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetFixedImage(target)
    elastixImageFilter.SetMovingImage(source)

    if params is not None:
        elastixImageFilter = set_filter_parameters(elastixImageFilter, params)

    result = elastixImageFilter.Execute()

    return result, elastixImageFilter
    

def set_filter_parameters(filter, params): 
    """
    If elastix relevant parameters are defined, 
    analyze them and set the filter to consider the params.

    """
    if "parameter_files" in params:

        parameter_files = params["parameter_files"]
        
        # Generate pmap based on list or file type provided
        if isinstance(parameter_files, list):
            pmap = sitk.VectorOfParameterMap()
            for parameter_file in parameter_files:
                pmap.append(filter.ReadParameterFile(parameter_file))
                
        else:
            pmap = filter.ReadParameterFile(params["parameter_file"])
            
        filter.SetParameterMap(pmap)

    if "target_mask" in params:
        filter.SetFixedMask(params["target_mask"])     

    if "source_mask" in params:
        filter.SetMovingMask(params["source_mask"]) 

    return filter


def get_registration_transform(fixed_image,
                               moving_image,
                               registration_type=TRANSFORMATION_MAP["Affine"]):
    """Performs the registration and returns a SimpleITK's `Transform` class which can be
    used to resample an image so that it is registered to another one. However, in our code
    we do not resample images but only use this information to find where the `moving_image` 
    should be truncated so that it contains only the part of the body that is found in the `fixed_image`. 
    Registration parameters are hardcoded and picked for the specific task of  CBCT to CT translation. 
    TODO: consider making the adjustable in config.
    
    
    Parameters:
    ------------------------
    fixed_image:
    moving_image: 
    registration_type: Type of transformation to be applied to the moving image
    http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/22_Transforms.html
    
    """

    # Get seed from environment variable if set for registration randomness
    seed = int(os.environ.get('PYTHONHASHSEED')
               ) if 'PYTHONHASHSEED' in os.environ else sitk.sitkWallClock

    # SimpleITK registration's supported pixel types are sitkFloat32 and sitkFloat64
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=200)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)

    registration_method.SetMetricSamplingPercentage(0.01, seed)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Align the centers of the two volumes and set the center of rotation to the center of the fixed image
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, registration_type,
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetNumberOfThreads(1)

    final_transform = registration_method.Execute(fixed_image, moving_image)
    return final_transform
