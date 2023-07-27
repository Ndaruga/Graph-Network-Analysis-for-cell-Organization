import os
import pandas as pd
import pprint

ROOT_DIR = os.path.join('data', 'sliced_images')


# -------------------READ FILE NAMES AS PANDAS DATA FRAAME-------------------------#

def get_image_filenames():
    """Gets a data frame of image filenames.

    Returns:
        A data frame of image filenames.
    """

    image_filenames = []
    mask_filenames = []
    label_filenames = []

    for dir_name in os.listdir(ROOT_DIR):
        for filename in os.listdir(os.path.join(ROOT_DIR, dir_name)):
            if filename.endswith("_image.png"):
                image_filenames.append(str(os.path.join(dir_name, filename)))
            elif filename.endswith("_mask.png"):
                mask_filenames.append(str(os.path.join(dir_name, filename)))
            elif filename.endswith("_label.tif"):
                label_filenames.append(str(os.path.join(dir_name, filename)))

    df = pd.DataFrame({
        "images": image_filenames,
        "masks": mask_filenames,
        "labels": label_filenames,
    })
    df.to_csv('filenames.csv', index=True, header=True)
    return df

get_image_filenames()
