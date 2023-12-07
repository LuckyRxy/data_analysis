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
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import SimpleITK as sitk
from radiomics import featureextractor
from PyCode_PyRadiomicsVGG.NiftyIO import readNifty

from radiomics import setVerbosity

setVerbosity(60)


def ShiftValues(image, value):
    image = image + value
    print("Range after Shift: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image


def SetRange(image, in_min, in_max):
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (in_max - in_min) + in_min

    image[image < 0] = 0
    image[image > image.max()] = image.max()
    print("Range after SetRange: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image


def SetGrayLevel(image, levels):
    # array's values between 0 & 1
    image = image * levels
    image = image.astype(np.uint8)  # get into integer values
    print("Range after SetGrayLevel: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image


def saveXLSX(filename, df):
    # write to a .xlsx file.

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer._save()


def GetFeatures(featureVector, i, patient_id, nodule_id, diagnosis):
    new_row = {}
    # Showing the features and its calculated values
    for featureName in featureVector.keys():
        # print("Computed {}: {}".format(featureName, featureVector[featureName]))
        if ('firstorder' in featureName) or ('glszm' in featureName) or \
                ('glcm' in featureName) or ('glrlm' in featureName) or \
                ('gldm' in featureName) or ('shape' in featureName):
            new_row.update({featureName: featureVector[featureName]})
    lst = sorted(new_row.items())  # Ordering the new_row dictionary
    # Adding some columns  
    lst.insert(0, ('diagnosis', diagnosis))
    lst.insert(0, ('slice_number', i))
    lst.insert(0, ('nodule_id', nodule_id))
    lst.insert(0, ('patient_id', patient_id))
    od = OrderedDict(lst)
    return od


def SliceMode(patient_id, nodule_id, diagnosis, image, mask, meta1, meta2, extractor, maskMinPixels=200):
    myList = []
    i = 0

    while i < image.shape[2]:  # X, Y, Z
        # Get the axial cut
        img_slice = image[:, :, i]
        mask_slice = mask[:, :, i]
        try:
            if maskMinPixels < mask_slice.sum():
                # Get back to the format sitk
                img_slice_sitk = sitk.GetImageFromArray(img_slice)
                mask_slice_sitk = sitk.GetImageFromArray(mask_slice)

                # Recover the pixel dimension in X and Y
                (x1, y1, z1) = meta1.spacing
                (x2, y2, z2) = meta2.spacing
                img_slice_sitk.SetSpacing((float(x1), float(y1)))
                mask_slice_sitk.SetSpacing((float(x2), float(y2)))

                # Extract features
                featureVector = extractor.execute(img_slice_sitk,
                                                  mask_slice_sitk,
                                                  voxelBased=False)
                od = GetFeatures(featureVector, i, patient_id, nodule_id, diagnosis)
                myList.append(od)
            # else:
            #     print("features extraction skipped in slice-i: {}".format(i))
        except:
            print("Exception: skipped in slice-i: {}".format(i))
        i = i + 1

    df = pd.DataFrame.from_dict(myList)
    return df


def get_file_path(directory):
    file_paths = []
    files = os.listdir(directory)
    for file in files:
        file_paths.append(file)

    return file_paths


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


#### Parameters to be configured
db_path = r'D:\Project\PythonProject\data_analysis\resources\VOIs'
imageDirectory = 'image'
maskDirectory = 'nodule_mask'
imageNames = get_file_path(os.path.join(db_path, imageDirectory))
maskNames = get_file_path(os.path.join(db_path, imageDirectory))
# imageName = os.path.join(db_path, imageDirectory, 'LIDC-IDRI-0003_R_3.nii.gz')
# maskName = os.path.join(db_path, maskDirectory, 'LIDC-IDRI-0003_R_3.nii.gz')
####


# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.
params = 'config/Params.yaml'

# Initializing the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(params)

###
# patient_id = 'LIDC-IDRI-0003'
# nodule_id = 2
#
# df_mv = pd.read_excel(r'D:\Project\PythonProject\data_analysis\resources\MetadatabyNoduleMaxVoting.xlsx',
#                       sheet_name='ML4PM_MetadatabyNoduleMaxVoting',
#                       engine='openpyxl'
#                       )
#
# diagnosis = df_mv[(df_mv.patient_id == patient_id) & (df_mv.nodule_id == nodule_id)].Diagnosis_value.values[0]
# # print((df_mv.patient_id == patient_id) & (df_mv.nodule_id == nodule_id))
# print(df_mv[(df_mv.patient_id == patient_id) & (df_mv.nodule_id == nodule_id)])
# print(f'diagnosis:{diagnosis}')
###

# 读取已存在的 Excel 文件
existing_file_path = r"D:\Project\PythonProject\data_analysis\PyCode_PyRadiomicsVGG\features.xlsx"  # 替换为实际的文件路径
existing_data = pd.read_excel(existing_file_path)

# Reading image and mask
i = 0
while i < 300:
    patient_id = re.search(r'([A-Z0-9-]+)_R_\d+', imageNames[i]).group(1)
    nodule_id = int(re.search(r'R_(\d+)', imageNames[i]).group(1))
    print(f'patient_id:{patient_id}',f'nodule_id:{nodule_id}')
    df_mv = pd.read_excel(r'D:\Project\PythonProject\data_analysis\resources\MetadatabyNoduleMaxVoting.xlsx',
                          sheet_name='ML4PM_MetadatabyNoduleMaxVoting',
                          engine='openpyxl'
                          )
    diagnosis = df_mv[(df_mv.patient_id == patient_id) & (df_mv.nodule_id == nodule_id)].Diagnosis_value.values[0]
    print(f'diagnosis:{diagnosis}')


    image, meta1 = readNifty(os.path.join(db_path, imageDirectory, imageNames[i]), CoordinateOrder='xyz')
    mask, meta2 = readNifty(os.path.join(db_path, maskDirectory, maskNames[i]), CoordinateOrder='xyz')
    ### PREPROCESSING
    image = ShiftValues(image, value=1024)
    image = SetRange(image, in_min=0, in_max=4000)
    image = SetGrayLevel(image, levels=24)
    # Extract features slice by slice.
    df = SliceMode(patient_id, nodule_id, diagnosis, image, mask, meta1, meta2, extractor, maskMinPixels=200)
    # print(f"df:{df}")

    # 调用函数附加新数据到已存在的 Excel 文件
    append_data_to_excel(existing_file_path, df)
    i += 1

### PREPROCESSING
# image = ShiftValues(image, value=1024)
# image = SetRange(image, in_min=0, in_max=4000)
# image = SetGrayLevel(image, levels=24)

# Extract features slice by slice.
# df = SliceMode(patient_id, nodule_id, diagnosis, image, mask, meta1, meta2, extractor, maskMinPixels=200)

# if you get this message: "ModuleNotFoundError: No module named 'xlsxwriter'"
# then install it doing this: pip install xlsxwriter
# saveXLSX('features.xlsx', df)
