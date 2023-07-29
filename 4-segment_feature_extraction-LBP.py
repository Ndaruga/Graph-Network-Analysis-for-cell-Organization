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
# print(df.head)
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
        num_features = 14  # 4 shape features, 8 LBP features, mean_intensity, std_intensity
        return [0.0] * num_features

    # Shape Features
    props = regionprops(labeled_components)[0]
    area = props.area
    perimeter = props.perimeter
    eccentricity = props.eccentricity
    solidity = props.solidity

    # Texture Features - Using Local Binary Patterns (LBP) as an example
    lbp = feature.local_binary_pattern(component, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 9), range=(0, 9))
    lbp_features = hist_lbp.astype(float)

    # Intensity Features
    mean_intensity = component.mean()
    std_intensity = component.std()

    # Combine all features into a single feature vector
    features = [area, perimeter, eccentricity, solidity]
    features.extend(lbp_features)
    features.extend([mean_intensity, std_intensity])

    return features


for index, row in df.iterrows():
    image_filename = row['images']
    mask_filename = row['masks']
    label = row['labels']

    # Read image and mask using skit
    image = io.imread(os.path.join(ROOT_DIR, image_filename))
    mask = io.imread(os.path.join(ROOT_DIR, mask_filename))

    # Apply the mask to the image to extract the region of interest
    segmented_component = cv2.bitwise_and(image, image, mask=mask)
    features = extract_features(segmented_component)

    if 'labels' in df.columns:
        features.append(row['labels'])

    features_list.append(features)

#---------------------GET PANDAS DATAFRAME-------------------------#

# Convert the features_list to a pandas DataFrame
feature_columns = ['area', 'perimeter', 'eccentricity', 'solidity']
lbp_columns = [f'lbp_{i}' for i in range(8)]  # LBP features
intensity_columns = ['mean_intensity', 'std_intensity']

print('\n\n'+'*'*30 + " FEATURES DATAFRAME " + '*'*30)
features_df = pd.DataFrame(features_list, columns=feature_columns + lbp_columns + intensity_columns + ['label'])
print(features_df.tail(15))
print("\nshape: ", features_df.shape)
features_df.to_csv('extracted_features.csv')

# print('\n\n'+'*'*20 + "STANDARDIZED DATAFRAME" + '*'*20)

# Standardize the features
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features_df.drop(columns=['label']))
# features_df[feature_columns + lbp_columns + intensity_columns] = scaled_features
# print(features_df)
