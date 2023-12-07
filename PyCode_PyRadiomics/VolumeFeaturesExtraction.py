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


def append_data_to_excel(existing_file, new_data, sheet_name='Sheet1', index=False):
    """
    在已存在的 Excel 文件中附加新数据。

    参数:
    - existing_file: 字符串，已存在的 Excel 文件的路径。
    - new_data: DataFrame，包含要附加的新数据。
    - sheet_name: 字符串，要附加数据的工作表名，默认为'Sheet1'。
    - index: 布尔值，是否包含索引列，默认为 False。
    """
    try:
        # 读取已存在的 Excel 文件
        existing_data = pd.read_excel(existing_file, sheet_name=sheet_name)

        # 附加新数据到已存在的数据下面
        merged_data = existing_data._append(new_data, ignore_index=True)

        # 将合并后的数据写入 Excel 文件
        merged_data.to_excel(existing_file, index=index, sheet_name=sheet_name)

        print("数据附加成功。")
    except Exception as e:
        print(f"发生错误：{e}")


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
