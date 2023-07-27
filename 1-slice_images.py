import os
import sys
import cv2
import tifffile
import matplotlib.pyplot as plt

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
input_labels_folder = os.path.join(ROOT_DIR, 'label')
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

# ------------List all the images and label files in the input images/labels folder --------------------#

image_files = [f for f in os.listdir(input_images_folder) if f.endswith('.png')]
label_files = [f for f in os.listdir(input_labels_folder) if f.endswith('.tif')]

for filename in image_files:
    input_image_path = os.path.join(input_images_folder, filename)
    input_mask_path = os.path.join(input_masks_folder, filename)  # Assuming the mask filenames are the same as the image filenames
    input_label_path = os.path.join(input_labels_folder, os.path.splitext(filename)[0] + '.tif')  # Corresponding label file path
    output_subfolder = os.path.splitext(filename)[0]  # Create a subfolder for each image
    output_subfolder_path = os.path.join(output_folder_path, output_subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)

    # Load the image, its corresponding mask, and the label using OpenCV and tifffile
    image = cv2.imread(input_image_path)
    mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
    label = tifffile.imread(input_label_path)

    # Partition the image, mask, and label into smaller tiles
    image_tiles = partition_image(image, tile_size)
    mask_tiles = partition_image(mask, tile_size)
    label_tiles = partition_image(label, tile_size)

    # Save each tile as a separate image, mask, and label
    for i, (image_tile, mask_tile, label_tile) in enumerate(zip(image_tiles, mask_tiles, label_tiles)):
        output_image_path = os.path.join(output_subfolder_path, f'{i}_image.png')
        output_mask_path = os.path.join(output_subfolder_path, f'{i}_mask.png')
        output_label_path = os.path.join(output_subfolder_path, f'{i}_label.tif')
        cv2.imwrite(output_image_path, image_tile)
        cv2.imwrite(output_mask_path, mask_tile)
        tifffile.imsave(output_label_path, label_tile)

print("\nSLICING COMPLETE\n")


#------------------- Visualize the images ------------------------#
print("\n#------------------- Visualize the images ------------------------#\n")

def visualize_images(images_folder, index:int):
    """
    Visuslize_images: Visualize a slice of image, mask and label
    
    Args:
        images_folder: folder containing the images
        index: slice index

    """
    print(f"visualizing image {images_folder} at slice {index}")

    image = cv2.imread(os.path.join(ROOT_DIR, 'sliced_images', images_folder, f"{index}_image.png"))
    mask = cv2.imread(os.path.join(ROOT_DIR, 'sliced_images', images_folder, f"{index}_mask.png"), cv2.IMREAD_GRAYSCALE)
    label = tifffile.imread(os.path.join(ROOT_DIR, 'sliced_images', images_folder, f"{index}_label.tif"))


    # Create a 1x3 subplot for each image, mask, and label
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Show the image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image")
    axes[0].axis('off')

    # Show the mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    # Show the label
    axes[2].imshow(label, cmap='jet')  # Use 'jet' colormap for better visualization of labels
    axes[2].set_title("Label")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

for i in range(5):
    visualize_images("ID4_A1_Regione-0-crop_slice_4", i)