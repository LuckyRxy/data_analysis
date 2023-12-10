"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
# Pyhton standard Visualization Library
import matplotlib.pyplot as plt
# Pyhton standard IOs Library
import os
# Basic Processing
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
# from scipy.ndimage import filters as filt
from scipy.ndimage import gaussian_filter as gfilt
from scipy.ndimage import median_filter as mfilt
### IMPORT SESSION FUNCTIONS
#### Session Code Folder
SessionPyFolder='D:\Project\PythonProject\data_analysis\PyCode_BasicSegmentation'
os.chdir(SessionPyFolder) #Change Dir 2 load session functions
# .nii Read Data
from NiftyIO import readNifty
# Volume Visualization
from VolumeCutBrowser import VolumeCutBrowser

######## LOAD DATA

#### Data Folders
SessionDataFolder=r"D:\Project\PythonProject\data_analysis\resources"
os.chdir(SessionDataFolder)


CaseFolder='VOIs'
NiiFile='LIDC-IDRI-0003_R_2.nii.gz'


#### Load Intensity Volume
NiiFile=os.path.join(SessionDataFolder,CaseFolder,'image',NiiFile)
niiROI,niimetada=readNifty(NiiFile)


######## VISUALIZE VOLUMES

### Interactive Volume Visualization 
# Short Axis Cuts
VolumeCutBrowser(niiROI)


######## SEGMENTATION PIPELINE

### 1. PRE-PROCESSING
# 1.1 Gaussian Filtering
sig=1
# niiROIGauss = gfilt.gaussian_filter(niiROI, sigma=sig)
niiROIGauss = gfilt(niiROI, sigma=sig)
# 1.2 MedFilter
sze=3
# niiROIMed = mfilt.median_filter(niiROI, sze)
niiROIMed = mfilt(niiROI, sze)
###

### 2. BINARIZATION (TH is Threshold)
Th = threshold_otsu(niiROI)
niiROISeg=niiROI>Th
# ROI Histogram
fig,ax=plt.subplots()
ax.hist(niiROI.flatten(),bins=50,edgecolor='k')      #直方图
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




