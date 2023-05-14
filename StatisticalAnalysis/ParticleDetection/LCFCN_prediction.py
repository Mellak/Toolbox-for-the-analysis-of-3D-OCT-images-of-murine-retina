'''
The code uses a pre-trained FCN8-VGG16 model to predict the segmentation of cells in images.
The training was done as in LCFCN, the ground truth for Uveitis particles are points on them.

The model takes as input a 3-channel RGB image and outputs a 2-channel image, where each channel represents the probability of a pixel belonging to a cell or the background.
The code then uses the predicted segmentation to identify the cells in the image and their centroids. The centroids are then saved to a CSV file.

Inputs  : A dataset of images and a pre-trained FCN8-VGG16 model.
Outputs : 2D images of masks of particle.
          A CSV containing the centroids of the cells in the images.
'''

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torchvision.transforms.functional as FT
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, RandomSampler
from torchvision import transforms, models
from torchsummary import summary
from torch.autograd import Variable, grad
from PIL import Image, ImageFont, ImageDraw
from skimage import morphology as morph
from skimage.measure import regionprops
from Models.FCN8_VGG16 import FCN8_VGG16
from Utilis import *
from datasets import transformers as ut
import skimage


def get_blobs(probs, roi_mask=None):
    probs = probs.squeeze()
    h, w = probs.shape
    pred_mask = (probs > 0.5).astype('uint8')
    blobs = np.zeros((h, w), int)
    blobs = morph.label(pred_mask == 1)
    if roi_mask is not None:
        blobs = (blobs * roi_mask[None]).astype(int)
    return blobs

def blobs2points(blobs):
    blobs = blobs.squeeze()
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs)
    assert points.ndim == 2
    for r in rps:
        y, x = r.centroid
        points[int(y), int(x)] = 1
    return points


def blobs2points(blobs):
    blobs = blobs.squeeze()
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs)
    assert points.ndim == 2
    for r in rps:
        y, x = r.centroid
        points[int(y), int(x)] = 1
    return points


torch.cuda.empty_cache()

universel_path = sys.argv[2]
name_of_retina = sys.argv[1] #'retina1' #
images_path    = universel_path + name_of_retina + '/png/'
save_path      = universel_path + name_of_retina + '/Particules_dir/'
model_path     = 'D:/DL_Project/Uveitis21Internship/StatisticalAnalysis/ParticleDetection/Weights/model_best.pth'

if not os.path.exists(save_path):
    os.makedirs(save_path)

model_name = 'base'

exp_dict = {"dataset": {'name': 'my_OCT',
                        'transform': 'rgb_normalize'},
            "model": {'name': 'lcfcn', 'base': "fcn8_vgg16"},
            "batch_size": 1,
            "max_epoch": 100,
            'dataset_size': {'train': 1, 'val': 1},
            'optimizer': 'adam',
            'lr': 1e-5
            }

model = FCN8_VGG16(n_classes=1).cuda()
model_data = torch.load(model_path)
model.load_state_dict(model_data['model'])

mean = [0, 0, 0]
std = [1, 1, 1]
transformer = ut.ComposeJoint(
    [
        [transforms.ToTensor(), None],
        [transforms.Normalize(mean=mean, std=std), None],
        [None, ut.ToLong()]
    ])

plot = False

for image_name in os.listdir(images_path):

    image_path = images_path + image_name
    # Read the image from disk.
    image_raw = cv2.imread(image_path)
    # Convert the image to a PIL image.
    collection = list(map(FT.to_pil_image, [image_raw, image_raw]))
    # Transform the image.
    image, _ = transformer(collection)
    # Create a batch of images.
    batch = {"images": image[None]}
    # Move the image to the GPU.
    image = image[None, :, :, :].cuda()


    # Predict the segmentation of the retina surface.
    pred_blobs = model(image).squeeze()
    pred_np_array = pred_blobs.cpu().detach().numpy()
    if plot:
        plt.imshow(pred_np_array)
        plt.show()

     
    blobs = get_blobs(probs=pred_np_array)
    if plot:
        plt.imshow(blobs)
        plt.show()
    # Save the prediction to a file.
    blobs[blobs != 0] = 255
    cv2.imwrite(save_path + image_name.replace('.png','.png'),blobs)

    if plot:
        # Create a figure with two subplots.
        f = plt.figure()
        # Add the prediction to the first subplot.
        f.add_subplot(1, 2, 1)
        plt.imshow(pred_np_array)
        # Add the blobs to the second subplot.
        f.add_subplot(1, 2, 2)
        plt.imshow(blobs)
        # Show the figure.
        plt.show(block=True)

        # Convert the blobs to points.
        points = blobs2points(blobs).squeeze()
        # Get the output points.
        output_points = np.transpose(np.nonzero(points > 0))

        # Create a DataFrame of the output points.
        df = pd.DataFrame({'image':image_name,'y_coord':output_points[:,0],'x_coord':output_points[:,1]})
        # Save the DataFrame to a CSV file.
        df.to_csv('MyOut.csv', mode='a', header=False,index=False)

