from pathlib import Path
from typing import Union

import SimpleITK as sitk
from clinical_evaluation.dicom_saver import DICOM_KEY_TAG, DICOM_TAG_KEY
from clinical_evaluation.dicom_saver import utils
import pydicom

import logging
logger = logging.getLogger(__name__)

class DicomSaver:
    def __init__(self, output_dir='.'):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image = None
        self.existing_metadata = None

    def save(self, image_path, dicom_path=None):
        self.image = utils.load_image(image_path)

        if dicom_path:
            self.existing_metadata = utils.load_dicom_metadata(dicom_path)
        
        self.dicom_tags = self.generate_dicom_series_metadata()
        self.save_dicom_slices()

    def generate_dicom_series_metadata(self):
        """
        Generate series level metadata for the created DICOM
        Refer for standard on how to do it:
        http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM


        """
        dicom_series_metadata = {}
        # Parameters that will be generated irrespective of metadata
        # exists from another DICOM or not.
        general_attribute_tags = utils.generate_general_attributes()
        dicom_series_metadata.update(general_attribute_tags)

        # Direction and slice thickness is computed 
        # from the image
        image_tags = utils.compute_tags_from_image(self.image)
        dicom_series_metadata.update(image_tags)

        # Generate and add UIDs 
        uid_dicom_tags = utils.generate_dicom_uids()
        dicom_series_metadata.update(uid_dicom_tags)

        # Copied from given DICOM metadata or absent
        if self.existing_metadata:
            cloned_metadata = utils.copy_dicom_metadata(dicom_series_metadata, self.existing_metadata)
            dicom_series_metadata.update(cloned_metadata)

        return dicom_series_metadata

            
    def save_dicom_slices(self):
        writer = sitk.ImageFileWriter()
        
        # Use the study/series/frame of reference information given in the meta-data
        # dictionary and not the automatically generated information from the file IO
        writer.KeepOriginalImageUIDOn()

        image_shape = self.image.GetSize()

        for slice_idx in range(image_shape[-1]):
            image_slice = self.image[:, :, slice_idx]

            slice_metadata = utils.generate_slice_metadata(self.image, slice_idx)
            self.dicom_tags.update(slice_metadata)

            utils.set_metadata_from_dict(image_slice, self.dicom_tags)

            slice_save_path = (self.output_dir / str(slice_idx)).with_suffix(".dcm")
            writer.SetFileName(str(slice_save_path))
            # Some readers fail to read compressed DICOMs
            writer.UseCompressionOff()
            logger.info(f"Saving slice: {slice_idx}")
            writer.Execute(image_slice)
            
            