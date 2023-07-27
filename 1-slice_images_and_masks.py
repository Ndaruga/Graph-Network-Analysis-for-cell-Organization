import os
import sys
import cv2

current_dir = os.getcwd()
sys.path.append(current_dir)
print(f"System Path: {sys.path}")


def check_dir(dir_name):
    """Checks if a directory exists and clears its contents, creates 
        it if it doesn' exist.

    Args:
        dir_name: The name of the directory to check.

    Returns:
        The directory name
    """

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        for file in os.listdir(dir_name):
            os.remove(os.path.join(dir_name, file))
    return dir_name


# -------------------------Variables-------------------------------------#

ROOT_DIR = "data"
tile_size = 256
input_images_folder = os.path.join(ROOT_DIR, 'image')
input_masks_folder = os.path.join(ROOT_DIR, 'bwmask')
output_folder_path = check_dir(os.path.join(ROOT_DIR, "sliced_images"))

# -------------------------Variables-------------------------------------#


def partition_image(image, tile_size):
    """
    Partition Image: Function to partition a large image into 
    smaller tiles 256 X 256

    Args:
        image: single image to be partitioned
        tile_size: size of each partition
    
    Return: tile partitons
    """

    height, width = image.shape[:2]
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
    return tiles

# ----------------------List all the image files in the input images folder --------------------#

image_files = [f for f in os.listdir(input_images_folder) if f.endswith('.png')]

for filename in image_files:
    input_image_path = os.path.join(input_images_folder, filename)
    input_mask_path = os.path.join(input_masks_folder, filename)  # Assuming the mask filenames are the same as the image filenames
    output_subfolder = os.path.splitext(filename)[0]  # Create a subfolder for each image
    output_subfolder_path = os.path.join(output_folder_path, output_subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)

    # Load the image and its corresponding mask using OpenCV
    image = cv2.imread(input_image_path)
    mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)

    # Partition the image and mask into smaller tiles
    image_tiles = partition_image(image, tile_size)
    mask_tiles = partition_image(mask, tile_size)

    # Save each tile as a separate image and mask
    for i, (image_tile, mask_tile) in enumerate(zip(image_tiles, mask_tiles)):
        output_image_path = os.path.join(output_subfolder_path, f'{i}_image.png')
        output_mask_path = os.path.join(output_subfolder_path, f'{i}_mask.png')
        cv2.imwrite(output_image_path, image_tile)
        cv2.imwrite(output_mask_path, mask_tile)

