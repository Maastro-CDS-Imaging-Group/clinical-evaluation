import SimpleITK as sitk

def hu_correction(image: sitk.Image, cval=-1024):
    """
    Offset CBCT values with certain cval
    """
    image = image + cval
    return image


