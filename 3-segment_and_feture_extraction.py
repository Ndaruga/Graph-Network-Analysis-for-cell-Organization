import cv2
import os
import pandas as pd
import numpy as np
from skimage import io, feature, color, measure
from skimage.measure import regionprops
# from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('filenames.csv')
ROOT_DIR = os.path.join('data', 'sliced_images')
features_list = []

# -------------------------------------------------------------------------#

def extract_features(component):
    """
    extract features from an individual segmented component

    Args:
        components: the extracted components
    return:
        Extracted features
    """

    if len(component.shape) == 3: # Convert the component to grayscale if needed
        component = color.rgb2gray(component)
    
    # Check if any regions were detected
    labeled_components = measure.label(component.astype(int))
    if labeled_components.max() == 0:
        num_features = 12  # 4 shape features, 8 LBP features, mean_intensity, std_intensity
        return [0.0] * num_features



    # Shape Features
    props = regionprops(component.astype(int))[0]
    area = props.area
    perimeter = props.perimeter
    eccentricity = props.eccentricity
    solidity = props.solidity

    # Texture Features - Using Local Binary Patterns (LBP)
    lbp = feature.local_binary_pattern(component, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 9), range=(0, 9))
    lbp_features = hist_lbp.astype(float)
    # print(lbp_features)

    # Intensity Features
    mean_intensity = component.mean()
    std_intensity = component.std()

    # Combine all features into a single feature vector
    features = [area, perimeter, eccentricity, solidity]
    features.extend(lbp_features)
    features.extend([mean_intensity, std_intensity])
    print(features)

    return features


for index, row in df.iterrows():
    image_filename = row['images']
    mask_filename = row['masks']
    label = row['labels']

    # Read image and mask using OpenCV
    image = cv2.imread(os.path.join(ROOT_DIR, image_filename))
    mask = cv2.imread(os.path.join(ROOT_DIR, mask_filename), cv2.IMREAD_GRAYSCALE)

    # Apply the mask to the image to extract the region of interest
    segmented_component = cv2.bitwise_and(image, image, mask=mask)
    # print(segmented_component)
    features = extract_features(segmented_component)
    features.append(row['labels'])
    # print(features)


#---------------------GET PANDAS DATAFRAME-------------------------#
feature_columns = ['area', 'perimeter', 'eccentricity', 'solidity']
lbp_cols = [f'lbp_{i}' for i in range(8)]
intensity_cols = ['mean_intensity', 'std_intesity']
cols = feature_columns + lbp_cols + intensity_cols + ['label']

# print('\n\n'+'*'*20 + " FEATURE DATAFRAME " + '*'*20)
# print(pd.DataFrame(features, columns=cols))

# print(features_df.head())

# standardize the features
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features_df.drop(columns=['labels']))
# features_df[feature_columns + lbp_columns + intensity_columns] = scaled_features

# print('\n\n'+'*'*20 + "STANDARDIZED DATAFRAME" + '*'*20)
# print(feature_df.head())


# Optionally, visualize the segmented component
# cv2.imshow('Segmented Component', segmented_component)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
