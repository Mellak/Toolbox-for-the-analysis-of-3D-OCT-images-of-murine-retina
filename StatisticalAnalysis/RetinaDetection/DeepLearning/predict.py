'''
This code performs Retina surface segmentation using a pre-trained U-Net.
The model predicts masks for input images, and saves the predicted masks as images.
    Input:
         pre-trained U-Net model and a dataset of images.
    Output:
        A mask that covers the retina.
'''


import numpy as np
import torch
from PIL import Image
import cv2
from unet import UNet
from dataset import RetinaDatasetTest
import matplotlib.pyplot as plt
import sys
import os 

name_of_retina = sys.argv[1]
universel_path = 'D:/DL_Project/3D_Project/Test_Software/'+name_of_retina
data_folder    = universel_path + '/png/'
model_path     = 'D:/DL_Project/3D_Project/UNet/model_chkpoints/unet-cell.pt'
masque_path    = universel_path + '/masques_png/'

if not os.path.exists(masque_path):
        os.makedirs(masque_path)

def predict(model_path, data_folder, plot=True):
    """Predicts the segmentation of retina surfaces in images using a pre-trained U-Net model.

    Args:
        model_path: The path to the pre-trained U-Net model.
        data_folder: The path to the dataset of images.

    Returns:
        The predicted segmentations of the retina surfaces in the images.
    """

    # Load the pre-trained U-Net model.
    model = UNet()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    # Load the dataset of images.
    retina_dataset = RetinaDatasetTest(data_folder, eval=True)

    # Iterate over the dataset of images.
    for i, (input, label) in enumerate(retina_dataset):

        # Predict the segmentation of the retina surface.
        output = model(input).permute(0, 2, 3, 1).squeeze().detach().numpy()

        # Convert the prediction to an image.
        input_array = input.squeeze().detach().numpy()
        output_array = np.argmax(output, axis=2) * 255
        input_img = Image.fromarray(input_array)
        output_img = Image.fromarray(output_array.astype(dtype=np.uint16)).convert('L')

        # Save the prediction to a file.
        label = label.replace('D:/DL_Project/3D_Project/UNet_Retina_Segmentation/data/mini_test\\','')
        output_img_path = f'{masque_path}{label}'
        cv2.imwrite(output_img_path, np.array(output_img))

        # Display the input image, the predicted segmentation, and the combined image.
        print(f"Processing image {i+1}/{len(retina_dataset)} - {label}")
        if plot:
            input_img.show()
            output_img.show()
            plt.imshow(output_img, cmap='gray')
            plt.show()

if __name__ == "__main__":
    predict(model_path, data_folder, False)
