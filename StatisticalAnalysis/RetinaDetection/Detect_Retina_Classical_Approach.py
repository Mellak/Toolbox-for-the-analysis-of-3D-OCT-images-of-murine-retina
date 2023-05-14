'''
The code extracts the retina from series of 2D image.
The code first reads in the the 512 slices of a retina image and then performs a series of steps to extract the retina surface:

    1-Mask out particles in the image. This is done by creating a mask that covers the particles and then subtracting the mask from the image.
    2-Divide the image into small blocks and scale the intensity of each block. This is done to reduce the noise in the image.
    3-Apply a Gaussian filter to the image to smooth it. This is done to further reduce the noise in the image.
    4-Extract connected components from the image. This is done to identify the different objects in the image.
    5-Filter out small connected components.
    6- morphological operations to smooth the image. 
    7-Find the edges of the image. This is done by applying a Canny edge detector to the image.
    8-Create a mask that covers the retina. This is done by creating a mask that covers the y-coordinates of the edges.
    9-Save the mask to a file. This is done by writing the mask to a file.

    Input:
        512 slices of a 3D image of a retina.
    Output:
        A mask that covers the retina.

!!Note!!: It is important to experiment with the parameters to get the best results for a specific retina.
          The best results will vary depending on the quality of the image and the specific retina being extracted.

'''

import numpy as np
import cv2
import os
import time
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import scipy.ndimage
import sys

def extract_retina(universel_path, name_of_retina, plot=True):
    print('We are extracting retina for', name_of_retina)
    # Set up paths for input and output directories
    images_path     = universel_path + name_of_retina + '/png/'
    save_clean_path = universel_path + name_of_retina + '/Clean_D14/'
    masque_path     = universel_path + name_of_retina + '/masques_png/'
    particules_path = universel_path + name_of_retina + '/Particules_dir/'

    # Create output directories if they don't exist
    if not os.path.exists(save_clean_path):
        os.makedirs(save_clean_path)

    if not os.path.exists(masque_path):
        os.makedirs(masque_path)

    for image_path in os.listdir(images_path):
        if True:
            start = time.time()

            # Read input image and particules image
            img = cv2.imread(images_path + image_path, cv2.IMREAD_GRAYSCALE)
            particules_img = cv2.imread(particules_path + image_path, cv2.IMREAD_GRAYSCALE)

            # Mask out particles in the image
            for y in range(particules_img.shape[0]):
                for x in range(particules_img.shape[1]):
                    if particules_img[y][x] != 0:
                        if y < 500:
                            img[y, x] = 0

            # Preprocess image by dividing into small blocks and scaling the intensity
            my_out_img = np.zeros(img.shape)
            for idx in range(128):
                small_img = img[:, 4 * idx:4 * (idx + 1)]
                my_out_img[:, 4 * idx:4 * (idx + 1)] = (small_img - small_img.min()) / small_img.max() * 255

            cv2.imwrite('my_out_img.png', my_out_img)
            my_out_img = cv2.imread('my_out_img.png', cv2.IMREAD_GRAYSCALE)

            # Apply Gaussian filter for smoothing
            kernel = np.ones((5, 5), np.float32) / 25
            gauss_img = cv2.filter2D(my_out_img, -1, kernel)
            mean_img = np.copy(gauss_img)
            weak_thresh = np.copy(gauss_img)
            gauss_img[gauss_img < 50] = 0
            gauss_img[gauss_img >= 50] = 255

            weak_thresh[weak_thresh < 50] = 0
            weak_thresh[weak_thresh >= 50] = 255

            end_y = 0
            end_x = 511
            start_x = 0

            # Extract connected components from the image
            nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gauss_img, connectivity=8)
            sizes = stats[1:, -1]
            nb_components = nb_components - 1

            min_size = 15000

            # Filter out small components
            gauss_img = np.zeros(gauss_img.shape)
            for i in range(0, nb_components):
                if sizes[i] >= min_size:
                    gauss_img[output == i + 1] = 255

            # Perform morphological operations for smoothing
            kernel = np.ones((7, 9), np.float32)
            weak_thresh = cv2.morphologyEx(weak_thresh, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((3, 3), np.float32)
            weak_thresh = cv2.morphologyEx(weak_thresh, cv2.MORPH_OPEN, kernel)

            final_column = gauss_img[:500, end_x]
            first_column = gauss_img[:500, start_x]

            end_y_array = np.argwhere(final_column != 0)
            end_y_array[end_y_array == 0] = 511

            start_y_array = np.argwhere(first_column != 0)
            start_y_array[start_y_array == 0] = 511

            kernel = np.ones((5, 5), np.float32) / 25
            dst = cv2.filter2D(gauss_img, -1, kernel)

            cv2.imwrite('Canny.png', dst)
            edges = cv2.Canny(cv2.imread('Canny.png', cv2.IMREAD_GRAYSCALE), 50, 50)

            ylist = []
            for in_x in range(512):
                point_y_limit = 1023
                while weak_thresh[point_y_limit, in_x] == 0 and point_y_limit > 1:
                    point_y_limit = point_y_limit - 1

                ypoints_of_x = np.argwhere(edges[:, in_x] != 0)
                max_diff = -1

                if len(ypoints_of_x) != 0:
                    out_y = ypoints_of_x[0]
                    for ypixel_point in ypoints_of_x:
                        y, x = np.squeeze(ypixel_point), in_x
                        wind_size = 21
                        if x < wind_size / 2 + 1:
                            small_down_window = mean_img[y:y + wind_size, x:x + wind_size]
                            small_up_window = mean_img[y - wind_size:y, x:x + wind_size]
                        elif x > (512 - wind_size / 2 - 1):
                            small_down_window = mean_img[y:y + wind_size, x:x + wind_size]
                            small_up_window = mean_img[y - wind_size:y, x:x + wind_size]
                        else:
                            small_down_window = mean_img[y:y + wind_size, x - int(wind_size / 2):x + int(wind_size / 2)]
                            small_up_window = mean_img[y - wind_size:y, x - int(wind_size / 2):x + int(wind_size / 2)]

                        small_down_mean = np.mean(small_down_window)
                        small_up_mean = np.mean(small_up_window)

                        diff = small_down_mean - small_up_mean

                        if diff > max_diff and abs(point_y_limit - ypixel_point > 250):
                            max_diff = diff
                            out_y = ypixel_point

                    ylist.append(out_y)
                else:
                    idx_dilat = 0
                    while len(ypoints_of_x) == 0:
                        kernel = np.ones((1, idx_dilat), np.uint8)
                        dilat = cv2.dilate(edges, kernel, iterations=1)
                        erod = cv2.erode(dilat, kernel, iterations=1)
                        ypoints_of_x = np.argwhere(erod[:, in_x] != 0)
                        idx_dilat += 1

                    out_y = ypoints_of_x[0]
                    for ypixel_point in ypoints_of_x:
                        y, x = np.squeeze(ypixel_point), in_x

                        wind_size = 21
                        if x < wind_size / 2 + 1:
                            small_down_window = mean_img[y:y + wind_size, x:x + wind_size]
                            small_up_window = mean_img[y - wind_size:y, x:x + wind_size]
                        elif x > (512 - wind_size / 2 - 1):
                            small_down_window = mean_img[y:y + wind_size, x:x + wind_size]
                            small_up_window = mean_img[y - wind_size:y, x:x + wind_size]
                        else:
                            small_down_window = mean_img[y:y + wind_size, x - int(wind_size / 2):x + int(wind_size / 2)]
                            small_up_window = mean_img[y - wind_size:y, x - int(wind_size / 2):x + int(wind_size / 2)]

                        small_down_mean = np.mean(small_down_window)
                        small_up_mean = np.mean(small_up_window)

                        diff = small_down_mean - small_up_mean

                        if diff > max_diff and abs(point_y_limit - ypixel_point > 250):
                            max_diff = diff
                            out_y = ypixel_point

                    ylist.append(out_y)

            yhat = scipy.ndimage.filters.median_filter(ylist, size=31)

            path = []
            path_2 = []
            for index in range(512):
                yx = np.array([np.squeeze(yhat[index]), index])
                path.append(yx)
                path_2.append(np.array([np.squeeze(ylist[index]), index]))

            masque_out_img = np.zeros_like(img)
            for ind_p in range(len(path)):
                point_y, point_x = path[ind_p]
                point_y_limit = 1023

                while weak_thresh[point_y_limit, point_x] == 0:
                    point_y_limit = point_y_limit - 1
                masque_out_img[point_y:point_y_limit, point_x] = 255

            if plot:
                plt.imshow(masque_out_img)
                plt.show()

            cv2.imwrite(masque_path + image_path, masque_out_img)

            real_img = cv2.imread(images_path + image_path)
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
            real_img[masque_out_img == 0] = 0
            cv2.imwrite(save_clean_path + image_path, real_img)

            print(image_path + " takes {} s".format(time.time() - start))

if __name__ == "__main__":
    name_of_retina = sys.argv[1]
    universal_path = sys.argv[2]
    extract_retina(universal_path, name_of_retina, False)