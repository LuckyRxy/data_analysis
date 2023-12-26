"""
This is the pipeline for the introduction to local image descriptors. 

Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""
### REFERENCES
# www.scipy-lectures.org/advanced/image_processing/#edge-detection

### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
import numpy as np
# Mayavi Visualization Functions
from mayavi import mlab
# Pyhton standard Visualization Library
import matplotlib

matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg

# Pyhton standard IOs Library
import os
import sys
# Basic Processing
from skimage.filters import threshold_otsu
import scipy.ndimage as filt
from scipy import ndimage as ndi

# Kmeans Clustering
from sklearn.cluster import KMeans

### IMPORT SESSION FUNCTIONS 
#### Session Code Folder
CodeMainDir = r'D:\Project\PythonProject\data_analysis\PyCode_LocalDescriptors'
sys.path.append(CodeMainDir)
# .nii Read Data
from PyCode_BasicSegmentation.NiftyIO import readNifty
######## PARAMETERS
#### Data Folders
SessionDataFolder = r'D:\Project\PythonProject\data_analysis\resources'
CaseFolder = 'CT'
ROIFolder = 'VOIs'
MaskROINiiFile = 'LIDC-IDRI-0001_R_1.nii.gz'
ROINiiFile = 'LIDC-IDRI-0001.nii.gz'
NiiFile = 'LIDC-IDRI-0016_GT1.nii.gz'

#### Processing Parameters

image_name = 'SAROI'  # filled_square, square, SA, SA_Mask, SAROI
sig = 0.1  # sigma of gaussian filter
Medsze = 3  # size of median filter
filter_image = 'gaussian'  # none, gaussian, median
gabor_params = 'default'  # default, non_default

######## LOAD DATA

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

### 4. FEATURE SPACES
## EX5: Compare Otsu binarization that only takes into account
#  image intensity to using the values of intensity together with
#  some of the local descriptors: (intensity,Laplacian)

### 4.0 Data
# SA Cut
k = int(niiROI.shape[2] / 2)  # Cut at the middle of the volume
im = niiROI[:, :, k]
im = (im - im.min()) / (im.max() - im.min())
imMask = niiMask[:, :, k]

# SA cut Laplacian
sx = filt.sobel(im, axis=1, mode='constant')
sxx = filt.sobel(sx, axis=1, mode='constant')
sy = filt.sobel(im, axis=0, mode='constant')
syy = filt.sobel(sy, axis=0, mode='constant')
Lap = sxx + syy

### 4.1 Intensity thresholding
Th = threshold_otsu(im)
imSeg = im > Th

# Show Segmentation
plt.figure()
plt.imshow(im, cmap='gray')
plt.contour(im, [Th], colors='red')
plt.show()

# Show Intensity histogram and Otsu Threshold
plt.figure()
plt.hist(im.flatten(), edgecolor='k', bins=5)
plt.hist(im[np.nonzero(imMask)], bins=5, edgecolor='k', alpha=0.5, facecolor='r', label='Lesion Values')
plt.plot([Th, Th], [0, 4000], 'k', lw=2, label='Otsu Threshold')
plt.legend()
plt.show()

### 4.2 Feature Space Partition
# Pixel Representation in a 2D space. In the plot
# each pixel it is assigned a x-coordinate given by its intensity
# and a y-coordinate given by its Laplacian
#              Pixel(i,j) ---> (im(i,j),Lap(i,j))
#
# EX6: Try to divide the plane with a line splitting (discriminating)
#      the red (lesion) and blue points (background)


plt.figure()
plt.plot(im.flatten(), Lap.flatten(), '.')
plt.plot(im[np.nonzero(imMask)], Lap[np.nonzero(imMask)], 'r.', label='Lesion Values')
plt.xlabel('Intensity')
plt.ylabel('Laplacian')
plt.title('Pixel Distribution in the Space of Values given by (Intensity,Laplacian).')
plt.legend()
plt.show()
print("!!!!!")

### 4.3 Kmeans Clustering
# EX7: Run k-means with and without normalization of the feature space

Lap = (Lap - Lap.min()) / (Lap.max() - Lap.min())
X = np.array((im.flatten(), Lap.flatten()))
kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X.transpose())
labels = kmeans.predict(X.transpose())

plt.figure()
plt.plot(im.flatten(), Lap.flatten(), '.')
plt.plot(im[np.nonzero(imMask)], Lap[np.nonzero(imMask)], 'r.', label='Lesion Values')
plt.xlabel('Intensity')
plt.ylabel('Laplacian')
plt.title('Pixel Distribution in the Space of Values given by (Intensity,Laplacian).')
x = X[0, np.nonzero(labels == 1)]
y = X[1, np.nonzero(labels == 1)]
plt.plot(x.flatten(), y.flatten(), 'k.', label='k-means Clustering')
plt.legend()
plt.show()
