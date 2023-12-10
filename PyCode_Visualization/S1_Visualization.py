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

### IMPORT SESSION FUNCTIONS
#### Session Code Folder (change to your path)
SessionPyFolder='D:\Project\PythonProject\data_analysis\PyCode_Visualization'
os.chdir(SessionPyFolder) #Change Dir 2 load session functions
# .nii Read Data
from PyCode_PyRadiomicsVGG.NiftyIO import readNifty
# Volume Visualization
from VolumeCutBrowser import VolumeCutBrowser


######## LOAD DATA

#### Data Folders (change to your path)
SessionDataFolder=r'D:\Project\PythonProject\data_analysis\resources'
os.chdir(SessionDataFolder)


CaseFolder='CT'
NiiFile='LIDC-IDRI-0003.nii.gz'


#### Load Intensity Volume(强度体积)
NiiFile=os.path.join(SessionDataFolder,CaseFolder,'image',NiiFile)
niivol,niimetada=readNifty(NiiFile)

#### Load Nodule Mask
NiiFile=os.path.join(SessionDataFolder,CaseFolder,'nodule_mask',NiiFile)
niimask,niimetada=readNifty(NiiFile)
print('niimask',niimask)
print('niimetada',niimetada)

######## VOLUME METADATA
print('Voxel Resolution (mm): ', niimetada.spacing)#体素分辨率
print('Volume origin (mm): ', niimetada.origen)#体积原点
print('Axes direction: ', niimetada.direction)


######## VISUALIZE VOLUMES
### Interactive Volume Visualization (交互式体积可视化)
# Short Axis View
VolumeCutBrowser(niivol)
VolumeCutBrowser(niivol, IMSSeg=niimask)
# Coronal View(冠状视图)
VolumeCutBrowser(niivol, Cut='Cor')
# Sagital View(矢状试图)
VolumeCutBrowser(niivol, Cut='Sag')

### Short Axis (SA) Image 
# Define SA cut
k=int(niivol.shape[2]/2) # Cut at the middle of the volume 
SA=niivol[:,:,k]#对三维数组进行切片，提取出一个二维平面

# Image
fig1=plt.figure()
plt.imshow(SA,cmap='gray')#使用灰度展示图像
plt.close(fig1) #close figure fig1

# Cut Level Sets
levels=[400]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111,aspect='equal') 
ax1.imshow(SA,cmap='gray')
plt.contour(SA,levels,colors='r',linewidths=2)
plt.close("all") #close all plt figures




