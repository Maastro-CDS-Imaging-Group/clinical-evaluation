from typing import Union
from pathlib import Path
import time

import SimpleITK as sitk

import uuid
from clinical_evaluation.dicom_saver import DICOM_KEY_TAG, DICOM_TAG_KEY

# ORG ROOT generated from https://www.medicalconnections.co.uk/
ORG_ROOT = "1.2.826.0.1.3680043.10.650"

def load_image(path: Union[Path, str]):
    try:
        return sitk.ReadImage(str(path))
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


def generate_dicom_uids():
    """
    UID generation: 
    #   http://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_B.html
    #   Each UID is composed of two parts, an <org root> and a <suffix>:
    #   UID = <org root>.<suffix>
    #   Org root generated from above is used with
    #   suffix added from UUID generation
    """
    uid_dict = {}
    uid_dict[DICOM_KEY_TAG.SeriesInstanceUID] = f"{ORG_ROOT}.{uuid.uuid1().int}"
    uid_dict[DICOM_KEY_TAG.StudyInstanceUID] = f"{ORG_ROOT}.{uuid.uuid1().int}"
    uid_dict[DICOM_KEY_TAG.StudyID] = f"{uuid.uuid1().int}"
    uid_dict[DICOM_KEY_TAG.ID] = f"{uuid.uuid1().int}"
    return uid_dict
    

def generate_general_attributes():
    """
    Generate general attributes: Description, Time and Date
    """
    general_attributes = {}
    general_attributes[DICOM_KEY_TAG.SeriesDescription] = "Deep Learning generated sCT"
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
    return metadata


def set_metadata_from_dict(image, input_dict):
    for tag, value in input_dict.items():
        image.SetMetaData(tag, value)