'''
This code takes the name of a retina (directory) as input and performs centroid detection on a labeled 3D NRRD volume (Output of 3D_CC.py).
It saves the particule images as PNG files (2D) and records the centroid coordinates in a CSV file.
The Particules_dir_cc2 directory contains 2D images of particles after beeing processed by 3D_CC.py
'''


import os
import pandas as pd
import numpy as np
import cv2
import nrrd
import skimage
import sys
from skimage import measure
from skimage.measure import regionprops



def detect_centroids(universel_path, name_of_retina):
    print('We are detecting centroids for', name_of_retina)

    # Read the labeled 3D particules - Output of 3D_CC algorithm
    cc_labels, header = nrrd.read(
        universel_path + name_of_retina + '/3D_images/labeled_volume.nrrd')
    cc_labels = np.transpose(cc_labels)
    print(cc_labels.shape)
    print(header)

    # Create a directory to save the particule images
    particules_dir = universel_path + name_of_retina + '/Particules_dir_cc2/'
    if not os.path.exists(particules_dir):
        os.makedirs(particules_dir)

    # Create a CSV file for saving the centroids
    csv_file = universel_path + name_of_retina + '/3D_images/MyCentroids.csv'
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print("Existing file deleted.")
    else:
        print("File does not exist.")
    my_header = True

    for index, img in enumerate(cc_labels):
        saved_particules = np.copy(img)
        saved_particules[saved_particules != 0] = 255
        saved_particules = saved_particules.astype('int')

        cv2.imwrite(particules_dir + str(index).zfill(4) + '.png', saved_particules)
        my_labels = np.unique(img[np.nonzero(img)])

        for label in my_labels:
            cpy_img = np.copy(img)
            cpy_img[cpy_img != label] = 0
            rps = regionprops(cpy_img)
            for r in rps:
                y, x = r.centroid
                df = pd.DataFrame(
                    {'image_idx': [str(index).zfill(4)], 'label': [label], 'y_coord': [int(y)], 'x_coord': [int(x)]})
                df.to_csv(csv_file, mode='a', header=my_header, index=False)
                my_header = False


# Example usage:
name_of_retina = sys.argv[1]
universel_path = sys.argv[2]
detect_centroids(universel_path, name_of_retina)
