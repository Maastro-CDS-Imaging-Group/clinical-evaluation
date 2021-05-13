import logging
from pathlib import Path

import SimpleITK as sitk
from clinical_evaluation.evaluation import metrics
from clinical_evaluation.registration import pipeline
from clinical_evaluation.utils import ops, preprocess
import matplotlib.pyplot as plt


def save_and_display_visuals(visuals, save_path):
    for key, visual in visuals.items():
        f = plt.figure(figsize=(15, 15))

        if "difference" in key:
            plt.imshow(visual,  cmap='coolwarm', vmin=-350, vmax=350)
            cb = plt.colorbar(orientation="horizontal",anchor=(1.0,0.0), ticks=[-350, 0, 350], fraction=0.03)
            cb.ax.tick_params(labelsize=30)

        else:
            plt.imshow(visual, cmap='gray')

        plt.axis('off')

        plt.savefig(f"{save_path}_{key}.png", pad_inches=0, bbox_inches='tight')


data_folder = Path("/home/suraj/Workspace/results/NKI/media_results/test_predictions").resolve()
patient = "21902070"
save_path = "/home/suraj/Workspace/results/NKI/media_results/paper_qualitative_images/supplementary/patient2"

data_folder = [folder for folder in sorted(data_folder.iterdir()) if folder.is_dir()]

for idx, model in enumerate(data_folder):
    folder = model / patient
    
    translated = folder / "translated.nrrd"
    deformed = folder / "deformed.nrrd"
    
    CT = sitk.ReadImage(str(deformed))
    CT = ops.make_isotropic(CT)
    CT = sitk.GetArrayFromImage(CT)

    
    if idx == 0:
        original = folder / "target.nrrd"
        CBCT = sitk.ReadImage(str(original))
        CBCT = ops.make_isotropic(CBCT)
        CBCT = sitk.GetArrayFromImage(CBCT)
        visuals = ops.get_visuals(CT, CBCT)
        save_and_display_visuals(visuals, f"{save_path}/Original")
        
        visuals = ops.get_visuals(CT, CT)
        save_and_display_visuals(visuals, f"{save_path}/Target")


    sCT = sitk.ReadImage(str(translated))
    sCT = ops.make_isotropic(sCT)
    sCT = sitk.GetArrayFromImage(sCT)

    visuals = get_visuals(CT, sCT)