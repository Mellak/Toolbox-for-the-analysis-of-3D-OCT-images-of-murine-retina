'''

This code measures the distance between the centroid of a particle and retina surface.

Inputs:
    A CSV file containing the centroids of the particles.
    A directory containing the masks of retinas.
Outputs:
    A CSV file containing the label of each particle and the distance between the centroid and the edge of the image.

'''

import os
import random
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.segmentation import watershed, find_boundaries
from skimage import morphology as morph

# Set the name of the directory
name_of_retina = sys.argv[1] #'retina1' #
universel_path = sys.argv[2]

# Define file paths
base_path              = universel_path + name_of_retina
original_path          = base_path + '/png/'
masques_path           = base_path + '/masques_png/'
centroids_particle_csv = base_path + '/3D_images/My_one_time_Centroids.csv'
to_save_file           = base_path + '/3D_images/New_Distance_File.csv'
# Read the CSV file
df = pd.read_csv(centroids_particle_csv)
print(df)

# Extract all labels
all_labels  = list(df['label'])
plot_things = False
for label in all_labels:
    resu = df.loc[df['label'] == label]
    image_idx = resu.iloc[0, 0]

    # Load the mask image
    msq_img = cv2.imread(os.path.join(masques_path, image_idx))
    msq_img = cv2.cvtColor(msq_img, cv2.COLOR_BGR2GRAY)

    # Convert the mask image to RGB
    new_img = cv2.cvtColor(msq_img, cv2.COLOR_GRAY2BGR)

    # Show the RGB mask image
    if plot_things:
        plt.imshow(new_img)
        plt.grid(False)
        plt.axis('off')
        plt.show()

    # Show the negative mask image
    if plot_things:
        plt.imshow(255 - new_img)
        plt.grid(False)
        plt.axis('off')
        plt.show()

    # Convert the mask image to grayscale
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # Compute the distance transform
    img_distance = ndimage.distance_transform_edt(255 - new_img)
    
    # Show the distance image
    if plot_things:
        plt.grid(False)
        plt.axis('off')
        plt.imshow(-img_distance)
        plt.pcolor(np.array(img_distance), cmap=plt.cm.viridis, vmin=0, vmax=300)
        plt.colorbar()
        plt.show()

    # Show the distance image without color mapping
    if plot_things:
        plt.imshow(img_distance)
        plt.show()

    # Calculate the distance at the specified coordinates
    distance = img_distance[resu.iloc[0, 2], resu.iloc[0, 3]]

    # Create a new DataFrame for distance and label
    new_df = pd.DataFrame({'label': [label], 'distance': [distance]})

    # Save the new DataFrame to CSV
    new_df.to_csv(to_save_file, mode='a', header=False, index=False)

    old_value = image_idx
