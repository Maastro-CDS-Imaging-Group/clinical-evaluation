from pathlib import Path

import imageio
import SimpleITK as sitk
from clinical_evaluation.utils import ops
import numpy as np

class RegistrationVisualizer:
    def __init__(self, outdir='out', save_mode='image'):
        self.save_mode = save_mode
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)

    def save_registration_visualizations(self, image1, image2, prefix=None, min_HU=-1000,\
                                            max_HU=2000, checkerboard=False, overlay=False):

        image1 = clip_intensities(image1, min_HU, max_HU)
        image2 = clip_intensities(image2, min_HU, max_HU)

        visualizations = {
            "image1": image1,
            "image2": image2,
        }

        if checkerboard:
            visualizations["checkerboard"] =  get_checkerboard_image(image1, image2)

        if overlay:
            visualizations["overlay"] =  get_overlay_differences(image1, image2)


        self.save_visuals(visualizations, prefix)

    def save_visuals(self, visualizations, prefix):
        if prefix is None:
            prefix = self.outdir.stem

        for key, visual in visualizations.items():
            path = (self.outdir / f"{prefix}_{key}")

            if self.save_mode in ["image", "image+video"]:
                path = path.with_suffix(".png")
                views = create_2d_views(visual)
                imageio.imwrite(path, views)

            if self.save_mode in ["video", "image+video"]:
                path = path.with_suffix(".mp4")
                frames = create_video(visual)
                imageio.mimwrite(path, frames)

            if self.save_mode == "axial":
                path = path.with_suffix(".png")
                array = sitk.GetArrayFromImage(visual)
                middle_axial = array[array.shape[0] // 2]
                imageio.imwrite(path, middle_axial)

            if self.save_mode == "all_mid_slices":
                path = path.with_suffix(".png")
                array = sitk.GetArrayFromImage(visual)
                middle_axial = array[array.shape[0] // 2]
                imageio.imwrite(path, middle_axial)                

            if self.save_mode == "axial_intervals":
                n_intervals = 6
                array = sitk.GetArrayFromImage(visual)
                axial_intervals = [int(array.shape[0] * (idx/n_intervals)) for idx in range(n_intervals)]

                for idx, interval in enumerate(axial_intervals):
                    axial = array[interval]
                    slice_path = f"{path}_slice_{idx}.png"
                    imageio.imwrite(slice_path, axial)   

            if self.save_mode == "all_intervals":
                n_intervals = 6
                visual = ops.make_isotropic(visual)
                array = sitk.GetArrayFromImage(visual)

                axial_intervals = [int(array.shape[0] * (idx/n_intervals)) for idx in range(n_intervals)]
                for idx, interval in enumerate(axial_intervals):
                    axial = array[interval]
                    slice_path = f"{path}_slice_axial_{idx}.png"
                    imageio.imwrite(slice_path, axial)

                sagittal_intervals = [int(array.shape[1] * (idx/n_intervals)) for idx in range(n_intervals)]
                for idx, interval in enumerate(sagittal_intervals):
                    sagittal = array[:, :, interval]
                    sagittal = np.flipud(sagittal)
                    slice_path = f"{path}_slice_sagittal_{idx}.png"
                    imageio.imwrite(slice_path, sagittal)

                coronal_intervals = [int(array.shape[2] * (idx/n_intervals)) for idx in range(n_intervals)]
                for idx, interval in enumerate(coronal_intervals):
                    coronal = array[:, interval]
                    coronal = np.flipud(coronal)
                    slice_path = f"{path}_slice_coronal_{idx}.png"
                    imageio.imwrite(slice_path, coronal)                    

def clip_intensities(image, wmin=None, wmax=None, min=0, max=255, type=sitk.sitkUInt8):
    image = sitk.Cast(
        sitk.IntensityWindowing(image,
                                windowMinimum=wmin,
                                windowMaximum=wmax,
                                outputMinimum=min,
                                outputMaximum=max), type)
    return image


def get_residual_images(image1, image2):
    return sitk.Subtract(image1, image2)

def get_checkerboard_image(image1, image2):
    filter = sitk.CheckerBoardImageFilter()
    return filter.Execute(image1, image2)

def get_overlay_differences(image1, image2):
    overlay_image = sitk.Compose(image1, image2, image1)
    return sitk.Cast(overlay_image, sitk.sitkVectorUInt8)

def create_2d_views(image):
    image = ops.make_isotropic(image)
    preview = ops.get_image_preview(image)
    return preview

def create_video(image):
    image = ops.make_isotropic(image)
    frames = ops.get_video_preview(image)
    return frames


if __name__ == "__main__":
    regviz = RegistrationVisualizer(save_mode='video')
    source = sitk.ReadImage("/home/suraj/Repositories/data/NKI/registered/21514736/source.nrrd")
    deformed = sitk.ReadImage("/home/suraj/Repositories/data/NKI/registered/21514736/deformed.nrrd")
    regviz.save_registration_visualizations(source, deformed)
