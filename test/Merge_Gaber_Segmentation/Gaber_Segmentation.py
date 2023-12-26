### 3. GABOR FILTERS
import os

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter as gfilt
from scipy.ndimage import median_filter as mfilt
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho

from test.Merge_Gaber_Segmentation.BrowseGaborFiltBank import BrowseGaborFiltBank
from test.Merge_Gaber_Segmentation.GaborFilters import GaborFilterBank2D
from test.Merge_Gaber_Segmentation.NiftyIO import readNifty
from test.Merge_Gaber_Segmentation.VolumeCutBrowser import VolumeCutBrowser

image_name = 'SAROI'  # filled_square, square, SA, SA_Mask, SAROI
sig = 0.1  # sigma of gaussian filter
Medsze = 3  # size of median filter
filter_image = 'gaussian'  # none, gaussian, median
gabor_params = 'default'  # default, non_default


#### Load ROI Volumes

# NiiPath=os.path.join(SessionDataFolder,CaseFolder,ROIFolder,ROINiiFile)
NiiPath = r'D:\Project\PythonProject\data_analysis\resources\VOIs\image\LIDC-IDRI-0001_R_1.nii.gz'
NiiPath = NiiPath.replace('\\', '/')
niiROI, _ = readNifty(NiiPath)

# NiiPath=os.path.join(SessionDataFolder,CaseFolder,ROIFolder,MaskROINiiFile)
NiiPath = r'D:\Project\PythonProject\data_analysis\resources\CT\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz'
NiiPath = NiiPath.replace('\\', '/')
niiMask, _ = readNifty(NiiPath)

# NiiPath=os.path.join(SessionDataFolder,CaseFolder,ROIFolder,NiiFile)
NiiPath = r'D:\Project\PythonProject\data_analysis\resources\CT\image\LIDC-IDRI-0001.nii.gz'
NiiPath = NiiPath.replace('\\', '/')
niivol, _ = readNifty(NiiPath)


VolumeCutBrowser(niiROI)


### Define Use Case Image

if image_name == 'filled_square':
    # Synthetic Filled Square
    im = np.zeros((256, 256))
    im[64:-64, 64:-64] = 1
    Angle = 15
    im = ndi.rotate(im, Angle, mode='constant')  # 2D rotation

elif image_name == 'square':
    # Synthetic Square
    im = np.zeros((256, 256))
    im[64:-64, 64:-64] = 1
    Angle = 15
    im = ndi.rotate(im, Angle, mode='constant')  # 2D rotation

    im = ndi.gaussian_gradient_magnitude(im, sigma=2)


elif image_name == 'SA':
    k = int(niivol.shape[2] / 2)  # Cut at the middle of the volume
    im = niivol[:, :, k]

elif image_name == 'SAROI':
    k = int(niiROI.shape[2] / 2)  # Cut at the middle of the volume
    im = niiROI[:, :, k]

elif image_name == 'SA_Mask':
    k = int(niiMask.shape[2] / 2)  # Cut at the middle of the volume
    im = niiMask[:, :, k]
else:
    raise Exception('Incorrect image name.')



# Filter Bank
if gabor_params == 'default':
    GaborBank2D_1, GaborBank2D_2, params = GaborFilterBank2D()
elif gabor_params == 'non_default':
    sigGab = [2, 4]
    freqGab = [0.25, 0.5]
    GaborBank2D_1, GaborBank2D_2, params = GaborFilterBank2D(sigma=sigGab, frequency=freqGab)

# print(f"GB2D_1:{GaborBank2D_1},GB2D_2:{GaborBank2D_2},params:{params}")
# print(np.sum(GaborBank2D_1))

# Show Filters
BrowseGaborFiltBank(GaborBank2D_1, params)
BrowseGaborFiltBank(GaborBank2D_2, params)

# Show Gaber Filter result(Mayavi)
# Gab2Show = 1
# fig1 = mlab.figure()
# mlab.surf(GaborBank2D_1[Gab2Show], warp_scale='auto')

# Apply Filters
# NFilt = np.shape(GaborBank2D_1)
# NFilt = NFilt[0]

NFilt = (
    len(GaborBank2D_1)
    if isinstance(GaborBank2D_1, list)
    else np.shape(GaborBank2D_1)[0]
)

# mlab.figure()
# # gabor_filter_index_for_3d_vis = 1
# mlab.surf(GaborBank2D_1[1], warp_scale="auto")
# mlab.title("3D Gabor Filter (Real Part)")
# mlab.show()  # cannot make this non-block. calling it optionally.

Ressze = np.concatenate((im.shape, np.array([NFilt])))
imGab1 = np.empty(Ressze)
imGab2 = np.empty(Ressze)
for k in range(NFilt):
    imGab1[:, :, k] = ndi.convolve(im, GaborBank2D_1[k], mode='wrap')
    imGab2[:, :, k] = ndi.convolve(im, GaborBank2D_2[k], mode='wrap')

VolumeCutBrowser(imGab1)
print(f"形状{imGab1.shape}")
# BrowseGaborFiltBank(GaborBank2D_1, params)
#
VolumeCutBrowser(imGab2)
# BrowseGaborFiltBank(GaborBank2D_2, params)


print('--------------------------------------------------------------------------------')


######## LOAD DATA

#### Data Folders
# SessionDataFolder=r"D:\Project\PythonProject\data_analysis\resources"
# os.chdir(SessionDataFolder)
#
#
# CaseFolder='VOIs'
# NiiFile='LIDC-IDRI-0003_R_2.nii.gz'
#
#
# #### Load Intensity Volume
# NiiFile=os.path.join(SessionDataFolder,CaseFolder,'image',NiiFile)
# niiROI,_=readNifty(NiiFile)

# print(f"形状{niiROI.shape}")

######## VISUALIZE VOLUMES

### Interactive Volume Visualization
# Short Axis Cuts
# VolumeCutBrowser(niiROI)


######## SEGMENTATION PIPELINE

### 1. PRE-PROCESSING
# 1.1 Gaussian Filtering
# sig=1
# # niiROIGauss = gfilt.gaussian_filter(niiROI, sigma=sig)
# niiROIGauss = gfilt(imGab1, sigma=sig)
#
# # 1.2 MedFilter
# sze=3
# # niiROIMed = mfilt.median_filter(niiROI, sze)
# niiROIMed = mfilt(imGab1, sze)
# ###
#
# VolumeCutBrowser(niiROIMed)

### 2. BINARIZATION (TH is Threshold)
Th = threshold_otsu(imGab1)
niiROISeg=imGab1>Th
# ROI Histogram
fig,ax=plt.subplots()
ax.hist(imGab1.flatten(),bins=50,edgecolor='k')      #直方图
# Visualize Lesion Segmentation
VolumeCutBrowser(niiROI,IMSSeg=niiROISeg)            #写错了应该是IMSSeg


### 3.POST-PROCESSING

# 3.1  Opening
szeOp=3
se=Morpho.cube(szeOp)
niiROISegOpen = Morpho.binary_opening(niiROISeg, se)

VolumeCutBrowser(niiROISegOpen,IMSSeg=None)

# 3.2  Closing
szeCl=3
se=Morpho.cube(szeCl)
niiROISegClose = Morpho.binary_closing(niiROISeg, se)

VolumeCutBrowser(niiROISegClose,IMSSeg=None)