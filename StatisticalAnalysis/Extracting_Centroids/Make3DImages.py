import os
import numpy as np
import nrrd
from PIL import Image
import sys

def organize_images_to_nrrd(folder_path, save_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get a list of all image files in the folder
    image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    # Check if the number of image files is not equal to 512
    if len(image_files) != 512:
        print(f"Invalid number of image files. Expected 512, but found {len(image_files)}.")
        return

    # Initialize an empty volume array
    volume = np.zeros((512, 1024, 512), dtype=np.uint8)

    # Load each image and insert it into the volume
    for index, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image_data = np.array(Image.open(image_path), dtype=np.uint8)

        # Check if the image size is not equal to (1024, 512)
        if image_data.shape != (1024, 512):
            print(
                f"Invalid image size. Expected (1024, 512), but found {image_data.shape}. Skipping image {index + 1}.")
            continue

        volume[index, :, :] = image_data

    # Save the volume as a NRRD file
    nrrd_path = os.path.join(folder_path, '3D_images')
    nrrd_name = save_path+'volume.nrrd'
    if not os.path.exists(nrrd_path):
        os.makedirs(nrrd_path)
    nrrd.write(os.path.join(nrrd_path, nrrd_name), volume)

    print("Organizing images into 3D NRRD image completed successfully.")

if __name__ == '__main__':
    # Example usage:
    name_of_retina = sys.argv[1] #'TX12_D6_C2R'
    universel_path = sys.argv[2]
    path2png_2D    = universel_path + name_of_retina + '/Particules_dir/'
    path2save      = universel_path + name_of_retina + '/3D_images/'
    organize_images_to_nrrd(path2png_2D, path2save)
