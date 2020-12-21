import SimpleITK as sitk

def get_abs_diff(source: sitk.Image, target: sitk.Image, mask: sitk.Image = None):

    if mask: 
        MaskImageFilter = sitk.MaskImageFilter()
        source = MaskImageFilter.Execute(source, mask)
        target = MaskImageFilter.Execute(target, mask)

    SubtractImageFilter = sitk.SubtractImageFilter()
    difference_image = SubtractImageFilter.Execute(source, target)

    AbsImageFilter = sitk.AbsImageFilter()
    abs_difference_image = AbsImageFilter.Execute(difference_image)
    return abs_difference_image


def get_statistics(image: sitk.Image):
    StatisticsImageFilter = sitk.StatisticsImageFilter()
    StatisticsImageFilter.Execute(image)
    mean = StatisticsImageFilter.GetMean()
    variance = StatisticsImageFilter.GetVariance()
    max = StatisticsImageFilter.GetMaximum()
    min = StatisticsImageFilter.GetMinimum()


    print(f"----- REPORT --------\n Mean: {mean} \n \
            Max: {max} \n Min: {min} \n Variance: {variance}")