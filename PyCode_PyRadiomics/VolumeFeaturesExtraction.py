"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""



import os
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor


from radiomics import setVerbosity

# from PyCode_PyRadiomics.featuresExtraction import get_file_path




def saveXLSX(filename, df):
    # write to a .xlsx file.

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer._save()

setVerbosity(60)


#### Parameters to be configured
db_path = r'D:\Project\PythonProject\data_analysis\resources\CT'
imageDirectory = 'image'
maskDirectory =  'nodule_mask'
imageName = os.path.join(db_path, imageDirectory, 'LIDC-IDRI-0001.nii.gz')
maskName  = os.path.join(db_path, maskDirectory, 'LIDC-IDRI-0001_R_1.nii.gz')
####
    

# Reading image and mask
imageITK = sitk.ReadImage(imageName)
maskITK = sitk.ReadImage(maskName)

# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.
params = 'config/FeaturesExtraction_Params.yaml'

# Initializing the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# Calculating features
featureVector = extractor.execute(imageITK, maskITK)


# Showing the features and its calculated values
for featureName in featureVector.keys():
    print("Computed {}: {}".format(featureName, featureVector[featureName]))
