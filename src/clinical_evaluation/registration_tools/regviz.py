from pathlib import Path

import imageio
import SimpleITK as sitk
from clinical_evaluation.registration_tools import utils
import numpy as np

class RegistrationVisualizer:
    def __init__(self, outdir='out', save_mode='image'):
        self.save_mode = save_mode
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)

    def save_registration_visualizations(self,
                                        image1,
                                        image2,
                                        min_HU=-1024,
                                        max_HU=2048):
        image1 = sitk.Cast(
            sitk.IntensityWindowing(image1,
                                    windowMinimum=min_HU,
                                    windowMaximum=max_HU,
                                    outputMinimum=0.0,
                                    outputMaximum=255.0), sitk.sitkUInt8)

        image2 = sitk.Cast(
            sitk.IntensityWindowing(image2,
                                    windowMinimum=min_HU,
                                    windowMaximum=max_HU,
                                    outputMinimum=0.0,
                                    outputMaximum=255.0), sitk.sitkUInt8)

        visualizations = { "checkerboard": self.get_checkerboard_image(image1, image2),
                            "duochrome": self.get_overlay_differences(image1, image2)}

        for key, visual in visualizations.items():
            path = (self.outdir / f"{self.outdir.stem}_{key}")

            if self.save_mode in ["image", "image+video"]:
                path = path.with_suffix(".png")
                views = self.create_2d_views(visual)
                imageio.imwrite(path, views)
                
            if self.save_mode in ["video", "image+video"]:
                path = path.with_suffix(".mp4")
                frames = self.create_video(visual)
                imageio.mimwrite(path, frames)

    def get_checkerboard_image(self, image1, image2, checker_pattern=[4, 4, 4]):
        filter = sitk.CheckerBoardImageFilter()
        return filter.Execute(image1, image2, checker_pattern)

    def get_overlay_differences(self, image1, image2):
        overlay_image = sitk.Compose(image1, image2, image1)
        return sitk.Cast(overlay_image, sitk.sitkVectorUInt8)

    def create_2d_views(self, image):
        image = utils.make_isotropic(image)
        preview = utils.get_image_preview(image)
        return preview

    def create_video(self, image):
        image = utils.make_isotropic(image)
        frames = utils.get_video_preview(image)
        return frames        


if __name__ == "__main__":
    regviz = RegistrationVisualizer(save_mode='video')
    image1 = sitk.ReadImage("/home/suraj/Repositories/data/NKI/registered/21514736/source.nrrd")
    image2 = sitk.ReadImage("/home/suraj/Repositories/data/NKI/registered/21514736/deformed.nrrd")
    regdict = regviz.get_registration_visualizations(image1, image2)