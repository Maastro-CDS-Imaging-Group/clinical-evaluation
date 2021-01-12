import logging
import time
from pathlib import Path
from typing import Union

import uuid
import SimpleITK as sitk
from clinical_evaluation.dicom_saver import DICOM_KEY_TAG, DICOM_TAG_KEY
from pydicom.uid import generate_uid

logger = logging.getLogger(__name__)
# ORG ROOT generated from https://www.medicalconnections.co.uk/
ORG_ROOT = '1.2.826.0.1.3680043.10.650.'
IGNORE_TAGS = ["RescaleSlope", "RescaleIntercept"]

def load_image(path: Union[Path, str]):
    try:
        image = sitk.ReadImage(str(path))
        image = sitk.Cast(image, sitk.sitkInt16)
        return image
    except Exception as e:
        raise ValueError(f"SITK could not read your image with exception :{e}")


def get_metadata_from_dicom(fn):
    """
    Read and return a dictionary of metadata from dicom file
    """
    metadata = {}
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(fn))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    for key in reader.GetMetaDataKeys():
        metadata[key] = reader.GetMetaData(key)

    logger.debug(f"Existing DICOM metadata: {metadata}")
    return metadata


def load_dicom_metadata(path: Union[Path, str]):
    """ 
    path: This can point to either a directory containing DICOM files or 
    a single dcm file. In case of directory, the first found dcm file will
    be used to extract tag values
    """
    path = Path(path).resolve()

    if path.is_dir():
        for fn in path.rglob('*.dcm'):
            return get_metadata_from_dicom(fn)
    else:
        return get_metadata_from_dicom(fn)


def compute_tags_from_image(image):
    """
    Tags that need to be computed explicitly from the image are added here.
    """
    image_computed_tags = {}
    # Image Orientation patient: Important for rendering of image
    direction = "\\".join([str(d) for d in image.GetDirection()])
    image_computed_tags[DICOM_KEY_TAG.ImageOrientationPatient] = direction
    # Slice thickness is set to same as z spacing. TODO: Make this more general
    image_computed_tags[DICOM_KEY_TAG.SliceThickness] = str(image.GetSpacing()[-1])

    logger.debug(f"Computed from Image \n Slice Thickness: \
         {image_computed_tags[DICOM_KEY_TAG.SliceThickness]} \
            ImageOrientationPatient: {image_computed_tags[DICOM_KEY_TAG.ImageOrientationPatient]}")

    return image_computed_tags


def generate_dicom_uids():
    """
    UID generation: 
    #   http://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_B.html
    #   Each UID is composed of two parts, an <org root> and a <suffix>:
    #   UID = <org root>.<suffix>
    #   Org root generated from above is used with
    #   suffix added from UUID generation

    generate_uid used from: 
    https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.uid.generate_uid.html
    """
    uid_dict = {}
    uid_dict[DICOM_KEY_TAG.SeriesInstanceUID] = generate_uid(prefix=ORG_ROOT)
    uid_dict[DICOM_KEY_TAG.FrameOfReferenceUID] = generate_uid(prefix=ORG_ROOT)
    uid_dict[DICOM_KEY_TAG.StudyInstanceUID] = generate_uid(prefix=ORG_ROOT)

    # StudyID limit is 16 characters
    uid_dict[DICOM_KEY_TAG.StudyID] = str(uuid.uuid4().int)[:16]
    uid_dict[DICOM_KEY_TAG.SeriesNumber] = str(1)
    uid_dict[DICOM_KEY_TAG.ID] = str(uuid.uuid4())
    return uid_dict
    

def generate_general_attributes():
    """
    Generate general attributes: Description, Time and Date
    """
    general_attributes = {}
    general_attributes[DICOM_KEY_TAG.SeriesDescription] = "Maastro CDS - Deep Learning Generated Image"
    general_attributes[DICOM_KEY_TAG.SeriesTime] = time.strftime("%H%M%S")
    general_attributes[DICOM_KEY_TAG.SeriesDate] = time.strftime("%Y%m%d")
    return general_attributes


def copy_dicom_metadata(generated_metadata, existing_metadata):
    """
    Copy metadata from existing DICOM file to generated metadata 
    while avoiding collision with existing tags and UIDs.

    """
    metadata = {}

    # Patient ID will be overwritten by the existing metadata
    generated_metadata.pop(DICOM_KEY_TAG.ID)

    for tag, value in existing_metadata.items():
        # Ignore tag if it has already been generated
        if tag in generated_metadata:
            continue
        
        # Do not copy any of the UIDS
        if "UID" in DICOM_TAG_KEY[tag]:
            continue
        
        # Intensity values are retained as is without any rescale parameters applied.
        if DICOM_TAG_KEY[tag] in IGNORE_TAGS:
            continue

        metadata[tag] = value

    return metadata


def generate_slice_metadata(image, slice_idx):
    """
    Generate slice specific metadata
    """
    metadata = {}
    # Slice specific tags.
    metadata[DICOM_KEY_TAG.InstanceCreationDate] = time.strftime("%Y%m%d")
    metadata[DICOM_KEY_TAG.InstanceCreationTime] = time.strftime("%H%M%S")

    # ImagePositionPatient attribute for each slice is determined by 
    # the physical location of the slice.
    ImagePositionPatient = '\\'.join([str(pos) for pos in image.TransformIndexToPhysicalPoint((0,0,slice_idx))])
    metadata[DICOM_KEY_TAG.ImagePositionPatient] = ImagePositionPatient
    metadata[DICOM_KEY_TAG.InstanceNumber] = str(slice_idx)

    logger.debug(f"Image Position Patient: {ImagePositionPatient}")
    return metadata


def set_metadata_from_dict(image, input_dict):
    for tag, value in input_dict.items():
        image.SetMetaData(tag, value)
