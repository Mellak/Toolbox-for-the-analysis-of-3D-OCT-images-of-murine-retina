'''
This code is used to analyze the spatial clustering of events in a 3D space.
The code first reads in a CSV file that contains the centroids of the events.
The centroids are then used to calculate the Ripley's K function, which is a measure of the spatial clustering of points.
The Ripley's K function is plotted for a range of radii.
The complete spatial randomness line is also plotted for comparison.
The complete spatial randomness line represents the expected distribution of points if they were randomly distributed.
The difference between the Ripley's K function and the complete spatial randomness line indicates the degree of clustering.

Input:
    A CSV file that contains the centroids of the particels. The CSV file should have the following columns:
        label: The number of particle (its label, oobtained by using 3D_CC.py).
        x: The x-coordinate of the centroid.
        y: The y-coordinate of the centroid.
        z: The z-coordinate of the centroid (slice number).

Output:
    A plot of the Ripley's K function and the complete spatial randomness line. The plot should have the following axes: 
'''


import os
import pandas as pd
import random
from random import seed
import csv
from PIL import Image
import PIL.ImageOps
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import sys
import cv2
import nrrd
import skimage
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
from skimage import morphology as morph
import operator
#TX12_D14_E2R
#TX12_D6_C1L
#TX12_D14_C1L
plt.rcParams["figure.figsize"] = (20,15)

def Select_centroids_w_distance(dfr,masques_path,min_dist_thresh = 0,max_dist_thresh = 500):
    xls = []
    yls = []
    zls = []
    for i in range(len(list(dfr['label']))):
        image_name = masques_path + str(dfr.iloc[i, 0])
        img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2GRAY)

        z = int(str(dfr.iloc[i, 0]).replace('.png', ''))
        y = int(str(dfr.iloc[i, 2]))
        x = int(str(dfr.iloc[i, 3]))

        img_y = 0
        while (img[img_y, x] == 0):
            img_y = img_y + 1

        distance = abs(img_y - y)

        if (distance < max_dist_thresh and distance >= min_dist_thresh):
            xls.append(x)
            yls.append(y)
            zls.append(z)
        # print(row)

    xs = np.array(xls)
    ys = np.array(yls)
    zs = np.array(zls)

    return xs,ys,zs


def Select_centroids_w_distance_with_csv(dfr,distance_df,min_dist_thresh = 0,max_dist_thresh = 500):
    xls = []
    yls = []
    zls = []
    for i in (list(dfr['label'])):
        if(i not in distance_df.iloc[: , 0].values):
            continue
        distance_values = distance_df.loc[distance_df.iloc[: , 0] == i]
        distance = distance_values.values.tolist()[0][1]
        if distance <= 0:
            continue
        selected_row = dfr.loc[dfr.iloc[: , 1] == i]
        z = int(str(selected_row.iloc[0, 0]).replace('.png', ''))
        y = int(str(selected_row.iloc[0, 2]))
        x = int(str(selected_row.iloc[0, 3]))
        if (distance < max_dist_thresh and distance >= min_dist_thresh):
            xls.append(x)
            yls.append(y)
            zls.append(z)
    xs = np.array(xls)
    ys = np.array(yls)
    zs = np.array(zls)
    return xs,ys,zs

import ripleyk
radii =[]
radii2 = []
radii3 = []

for i in range(35):
    t = 2*i + 1
    radii.append(t)
    radii2.append(((t) ** 2) * np.pi)
    radii3.append(((t) ** 3) *4* np.pi/3)

makers = ['o','v','s','p','*','d','+']
folders_path = 'D:/DL_Project/3D_Project/Test_Software/'
day_choosed  = '' #D2, D6 OR D7, D14
all_images_folders = os.listdir(folders_path)
my_list = []
sample_dict = {}

print(all_images_folders)
for thing in (all_images_folders):
    if(day_choosed in thing):
        if not os.path.exists(folders_path+thing+'/3D_images/New_Distance_File.csv'):
            continue
        tmp_pd = pd.read_csv(folders_path+thing+'/3D_images/New_Distance_File.csv')
        lenght = tmp_pd[tmp_pd.columns[0]].count()
        sample_dict[thing] = lenght
        my_list.append(thing)


sorted_dict = dict( sorted(sample_dict.items(), key=operator.itemgetter(1),reverse=True))
my_list = list(sorted_dict.keys())
all_images_Example = my_list[:8]
j = 0;max_x = -1;max_y = -1
legendLines = []
write=True

for name_of_dir in all_images_Example:
    print('name of directory',name_of_dir, 'number of events', sorted_dict[name_of_dir])
    path_csv     = folders_path + name_of_dir + '/3D_images/My_one_time_Centroids.csv'
    masques_path = folders_path + name_of_dir + '/masques_png/'
    distance_csv = folders_path + name_of_dir + '/3D_images/New_Distance_File.csv'

    df          = pd.read_csv(path_csv)
    distance_df = pd.read_csv(distance_csv)

    colors = ['g', 'r', 'b', 'y', 'm', 'k', 'c']
    title = ''
    max_x = max(radii)
    for i in range(1):
        xs, ys, zs = Select_centroids_w_distance_with_csv(df, distance_df)

        if (len(xs) == 0):
            continue
        k = ripleyk.calculate_ripley(radii, 512, d1=xs, d2=ys, d3=zs,boundary_correct=False, CSR_Normalise=False)
        if(write):
            plt.plot(radii, k,  label='3D k-ripley function for day '+day_choosed.replace('D','')+'.')
            plt.plot(radii, radii3,'c' + makers[-1], label='Complete Spatial Randomness.')
            write=False
        else:
            plt.plot(radii, k)
            plt.plot(radii, radii3,'c' + makers[-1])
        plt.axis(xmin=0, xmax=max_x, ymin=0, ymax=4*1e7)
    j += 1


plt.legend(loc='upper left', fontsize=40)
plt.xlabel("Radius (r)",fontsize=35)
plt.ylabel("Value of 3D k-Ripley function.",fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.savefig(folders_path+day_choosed+'.png')
plt.savefig(folders_path+day_choosed+'.pdf', transparent=True, bbox_inches='tight')
plt.show()







