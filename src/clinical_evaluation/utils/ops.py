import logging

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from clinical_evaluation.utils import ops

logger = logging.getLogger(__name__)

##### Visualization Operations ############################

def make_isotropic(image, interpolator=sitk.sitkLinear):
    '''
    # Source: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html

    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats 
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    '''
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / min_spacing)) for osz, ospc in zip(original_size, original_spacing)
    ]
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0, image.GetPixelID())


def get_image_preview(sitk_image, orientation='horizontal'):
    """Saves a preview image for a given SimpleITK object. In case of 3D image, saves
    an image combined of middle slices from axial, coronal and saggital views.
    """
    dims = sitk_image.GetDimension()

    if dims == 3:
        image = sitk.GetArrayFromImage(sitk_image)
        middle_axial = image[image.shape[0] // 2]
        middle_sagittal = image[:, :, image.shape[2] // 2]
        middle_coronal = image[:, image.shape[1] // 2]
        middle_sagittal = np.flipud(middle_sagittal)
        middle_coronal = np.flipud(middle_coronal)

        preview_image = padded_stack((middle_axial, middle_sagittal, middle_coronal), \
            orientation=orientation)

    else:
        logger.error("Image preview not implemented for 2D and 4D images")
        return  # TODO: implement for 2D and 4D
    return preview_image


def get_video_preview(sitk_image, orientation='vertical'):
    """Saves a preview image for a given SimpleITK object. In case of 3D image, saves
    an image combined of middle slices from axial, coronal and saggital views.
    """
    dims = sitk_image.GetDimension()

    if dims == 3:
        image = sitk.GetArrayFromImage(sitk_image)

        frames = []

        # Get a minimum over all indices to display in the video
        last_idx = min(image.shape[:2])

        for idx in range(last_idx):
            axial = image[idx]
            sagittal = image[:, :, idx]
            coronal = image[:, idx]
            sagittal = np.flipud(sagittal)
            coronal = np.flipud(coronal)

            preview_image = padded_stack((axial, sagittal, coronal), \
                orientation=orientation)

            frames.append(preview_image)

    else:
        logger.error("Image preview not implemented for 2D and 4D images")
        return  # TODO: implement for 2D and 4D

    return frames


def clip_and_normalize_intensities(image, wmin=None, wmax=None, min=0, max=255, type=sitk.sitkUInt8):
    """Clip values based on wmin and wmax and normalize them to range provided in min and max
    Type casting can also be done by providing a type

    """
    image = sitk.Cast(
        sitk.IntensityWindowing(image,
                                windowMinimum=wmin,
                                windowMaximum=wmax,
                                outputMinimum=min,
                                outputMaximum=max), type)
    return image


def get_residual_images(image1, image2):
    """Subtract image1 from image2
    """
    return sitk.Subtract(image1, image2)

def get_checkerboard_image(image1, image2):
    """Get a checkerboard image comparing image1 and image2 
    http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html#Checkerboard
    """
    filter = sitk.CheckerBoardImageFilter()
    return filter.Execute(image1, image2)

def get_overlay_differences(image1, image2):
    """Get the standard overlay image between image1 and image2. This is most 
    commonly used to verify registrations
    http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html#Combine-scalar-images-to-create-color-image
    """
    overlay_image = sitk.Compose(image1, image2, image1)
    return sitk.Cast(overlay_image, sitk.sitkVectorUInt8)

def create_2d_views(image):
    """
    Get an image with combined middle axial, sagittal and coronal views 
    for a particular `image`
    """
    image = make_isotropic(image)
    preview = get_image_preview(image)
    return preview

def create_video(image):
    """
    Get a video with all axial, sagittal and coronal views for a particular `image`
    """
    image = make_isotropic(image)
    frames = get_video_preview(image)
    return frames

def get_visuals(ref, pred, min_HU=-135, max_HU=215):
    """
    Generates different visuals that are useful for visual inspection such as:
    1. Difference axial image.
    2. Zoomed central axial slice.
    3. Sagittal views
    """
    visuals = {}
    windowed_ref = np.clip(ref, min_HU, max_HU)
    windowed_pred = np.clip(pred, min_HU, max_HU)
    
    difference_image = windowed_ref - windowed_pred
    
    difference_views = get_axial_sagittal_views(difference_image, min_HU, max_HU, normalize=False)
    visuals.update({f'difference_{k}': v for k, v in difference_views.items()})
        
    pred_views = get_axial_sagittal_views(windowed_pred, min_HU, max_HU)
    visuals.update(pred_views)
    
    return visuals


def get_axial_sagittal_views(image, min_HU, max_HU, zoom_factor=3, normalize=True):
    """
    Get axial and sagittal views along with zoomed axial and sagittal views. 
    zoom_factor: Gives the amount of zoom that must be provided during center crop and zoom
    """
    views = {}
    
    if normalize:
        image = image - min_HU / (max_HU - min_HU)

    center = [dim//2 for dim in image.shape]
    width = [dim//zoom_factor for dim in image.shape]
    
    crop_range = [[c - w//2, c + w//2] for c, w in zip(center, width)]
    zoomed_patch = image[crop_range[0][0]:crop_range[0][1], crop_range[1][0]:crop_range[1][1], crop_range[2][0]:crop_range[2][1]]
    views["zoomed_axial"] = zoomed_patch[zoomed_patch.shape[0]//2]
    views["zoomed_sagittal"] = np.flipud(zoomed_patch[:, :, zoomed_patch.shape[2]//2])

    views["axial"] = image[image.shape[0]//2]
    views["sagittal"] = np.flipud(image[:, :, image.shape[2]//2])
    return views

##### Image Manipulation Operations ############################

def apply_mask(image: sitk.Image, mask: sitk.Image):
    mask_filter = sitk.MaskImageFilter()
    mask_filter.SetOutsideValue(-1000)
    return mask_filter.Execute(image, mask)


def slice_image(sitk_image: sitk.Image, start=(0, 0, 0), end=(-1, -1, -1)):
    """"Returns the `sitk_image` sliced from the `start` index (x,y,z) to the `end` index.
    """
    size = sitk_image.GetSize()
    assert len(start) == len(end) == len(size)

    # replace -1 dim index placeholders with the size of that dimension
    end = [size[i] if end[i] == -1 else end[i] for i in range(len(end))]

    slice_filter = sitk.SliceImageFilter()
    slice_filter.SetStart(start)
    slice_filter.SetStop(end)
    return slice_filter.Execute(sitk_image)


##### Plotting Utils #################################


def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            nda = nda[nda.shape[0] // 2, :, :]

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3, 4):
            raise Runtime("Unable to show 3D-vector Image")

        # take a z-slice
        nda = nda[nda.shape[0] // 2, :, :, :]

    ysize = nda.shape[0]
    xsize = nda.shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

    t = ax.imshow(nda, extent=extent, interpolation=None)

    if nda.ndim == 2:
        t.set_cmap("gray")

    if (title):
        plt.title(title)


def myshow3d(img,
             xslices=[],
             yslices=[],
             zslices=[],
             title=None,
             margin=0.05,
             dpi=80):
    size = img.GetSize()
    img_xslices = [img[s, :, :] for s in xslices]
    img_yslices = [img[:, s, :] for s in yslices]
    img_zslices = [img[:, :, s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_null = sitk.Image([0, 0], img.GetPixelID(),
                          img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
        d += 1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen, d])
        #TO DO check in code to get Tile Filter working with vector images
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [
                    sitk.VectorIndexSelectionCast(s, i) for s in img_slices
                ]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
            img = sitk.Compose(img_comps)

    myshow(img, title, margin, dpi)


def show_mid_slices(image, slices=3, show_axis=(True, True, True)):
    image = ops.make_isotropic(image)
    mid = [dim // 2 for dim in image.GetSize()]
    slices += 1

    slices_range = []

    for idx, axis in enumerate(show_axis):
        if axis:
            slices_range.append([mid[idx], mid[idx] + slices])
        else:
            slices_range.append([])


    myshow3d(image, xslices=slices_range[0], \
                    yslices=slices_range[1],
                    zslices=slices_range[2])


##### Simplified Utilities ############################

def all_identical(sequence):
    """Check if all values of a list or tuple are identical."""
    return sequence.count(sequence[0]) == len(sequence)  # https://stackoverflow.com/a/3844948


def pad_to(arr, target, index):
    ndim = len(arr.shape)
    # Pad needs to be ndimx2 of the array
    pad_tuple = np.zeros((ndim, 2), dtype=int)
    pad_tuple[index][1] = target - arr.shape[index]
    pad_tuple = [list(v) for v in pad_tuple]

    return np.pad(arr, pad_tuple)


def padded_stack(arrays, orientation='vertical'):
    if orientation == 'vertical':
        # 1st index is the width, if vertical stacking needs to
        # be done, it will be stacked along the width
        index = 1
        stack_fn = np.vstack

    elif orientation == 'horizontal':
        # 0th index is the height, if horizontal stacking needs to
        # be done, it will be stacked along the height
        index = 0
        stack_fn = np.hstack

    dims = [arr.shape[index] for arr in arrays]

    if not all_identical(dims):
        arrays = [pad_to(arr, max(dims), index) for arr in arrays]

    return stack_fn(arrays)    
