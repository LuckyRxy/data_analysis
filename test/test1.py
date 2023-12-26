import matplotlib

import numpy as np

# from PyCode_PyRadiomics.featuresExtraction import ShiftValues, SetRange, SetGrayLevel

matplotlib.use('TkAgg')

from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
# from scipy.ndimage import filters as filt
from scipy.ndimage import gaussian_filter as gfilt
from scipy.ndimage import median_filter as mfilt

import matplotlib.pyplot as plt
from NiftyIO import readNifty
from test.VolumeCutBrowser import VolumeCutBrowser
#
# def ShiftValues(image, value):
#     image = image + value
#     print("Range after Shift: {:.2f} - {:.2f}".format(image.min(), image.max()))
#     return image
#
#
# def SetRange(image, in_min, in_max):
#     image = (image - image.min()) / (image.max() - image.min())
#     image = image * (in_max - in_min) + in_min
#
#     image[image < 0] = 0
#     image[image > image.max()] = image.max()
#     print("Range after SetRange: {:.2f} - {:.2f}".format(image.min(), image.max()))
#     return image
#
#
# def SetGrayLevel(image, levels):
#     # array's values between 0 & 1
#     image = image * levels
#     image = image.astype(np.uint8)  # get into integer values
#     print("Range after SetGrayLevel: {:.2f} - {:.2f}".format(image.min(), image.max()))
#     return image
#
image, _ = readNifty(r'D:\Project\PythonProject\data_analysis\resources\VOIs\image\LIDC-IDRI-0003_R_2.nii.gz', CoordinateOrder='xyz')
#
#
#
#
# ### 1. PRE-PROCESSING
# # 1.1 Gaussian Filtering
# sig=1
# # niiROIGauss = gfilt.gaussian_filter(niiROI, sigma=sig)
# niiROIGauss = gfilt(image, sigma=sig)
# # 1.2 MedFilter
# sze=3
# # niiROIMed = mfilt.median_filter(niiROI, sze)
# niiROIMed = mfilt(image, sze)
# ###
#
# ### 2. BINARIZATION (TH is Threshold)
# Th = threshold_otsu(image)
# niiROISeg=image>Th
# # ROI Histogram
# fig,ax=plt.subplots()
# ax.hist(image.flatten(),bins=50,edgecolor='k')      #直方图
# # Visualize Lesion Segmentation
# VolumeCutBrowser(niiROISeg,IMSSeg=niiROISeg)            #写错了应该是IMSSeg
#
#
# ### 3.POST-PROCESSING
#
# # 3.1  Opening
# szeOp=3
# se=Morpho.cube(szeOp)
# niiROISegOpen = Morpho.binary_opening(niiROISeg, se)
#
# VolumeCutBrowser(niiROISegOpen,IMSSeg=None)
#
# # 3.2  Closing
# szeCl=3
# se=Morpho.cube(szeCl)
# niiROISegClose = Morpho.binary_closing(niiROISeg, se)
#
# VolumeCutBrowser(niiROISegClose,IMSSeg=None)
#
#
#
# ## Apply the same preprocesing used in featuresExtraction.py
# ### PREPROCESSING
image = image[:,:,5]
#
#
# image = ShiftValues(image, value=1024)
# image = SetRange(image, in_min=0, in_max=4000)
# image = SetGrayLevel(image, levels=1)
#
fig1 = plt.figure()
# Th = threshold_otsu(image)
# image=image>Th
plt.imshow(image,cmap='gray')
plt.show()

# # VolumeCutBrowser(image,IMSSeg=None)


# 从 .npz 文件加载数据
import numpy as np

data = np.load('../resources/slice_glcm1d.npz', allow_pickle=True)

# 查看 .npz 文件中包含的数组的键
print("Keys in the NPZ file:", data.keys())
array_data = data['slice_features']

# 访问数组，这里假设有一个名为 'array_name' 的数组
# array_data = data['array_name']

data.close()

print(array_data)