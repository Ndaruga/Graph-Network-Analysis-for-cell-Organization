import os
import pandas as pd
import numpy as np
from skimage import io, color
from skimage.measure import regionprops
from skimage.feature import local_binary_pattern
from keras.models import load_model
import cv2

# Load the U-Net segmentation model
unet_model = load_model('unet_mmodel.h5')
df = pd.read_csv('filenames.csv')
# print(df.head)
ROOT_DIR = os.path.join('data', 'sliced_images')

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


features_list = []

for _, row in df.iterrows():
    image_filename = row['images']
    mask_filename = row['masks']
    label = row['labels']

    image = io.imread(image_filename)
    mask = io.imread(mask_filename)

    # Apply the mask to the image to extract the region of interest
    segmented_component = cv2.bitwise_and(image, image, mask=mask)
    resized_image = cv2.resize(segmented_component, (256, 256))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    # Perform cell segmentation using U-Net
    segmented_cells = unet_model.predict(input_image)[0]

    cell_labels = np.unique(segmented_cells)
    for label in cell_labels:
        cell_mask = (segmented_cells == label).astype(np.uint8)
        cell_image = cv2.bitwise_and(segmented_component, segmented_component, mask=cell_mask)
        features = extract_features(cell_image)
        if 'labels' in df.columns:
            features.append(row['labels'])

        features_list.append(features)
        print(features_list)

