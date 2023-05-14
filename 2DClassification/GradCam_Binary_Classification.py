import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from random import randrange, uniform
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
from haven import haven_utils as hu
from haven import haven_img as hi

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
    GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask
    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


def normalize_gradient_image(gradient):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()

    return gradient


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


# Data paths
day_data = 'Final_Multi_class_data'
universel_path = 'D:/DL_Project/Uveitis21Internship/2DClassification/'


# Dataset class
class ImageData(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.Image_name[index] + '.png'
        label = self.df.label[index]
        img_path = self.data_dir + img_name

        # Handle missing image files
        while not os.path.isfile(img_path):
            print(img_path)
            number = img_name.split("_", -1)[-1]
            number = number.replace('.png', '')
            irand = randrange(0, 512)
            img_name = img_name.replace(number, str(irand))
            img_path = self.data_dir + img_name

        src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        image = self.transform(image)

        return image, label, img_name


# Transformation for test data
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1]),
])

# Data loading and preprocessing
day = 'D14' # Can be D2, D6, D14
num_exp = 3
testdb = pd.read_csv(
    universel_path + 'Csv_files/train_test_' + day + '/' + 'test' + str(num_exp) + '_' + day + '.csv')
test_dir = '/home/youness/data/images/'
test_data = ImageData(df=testdb, data_dir=test_dir, transform=test_transform)
test_loader = DataLoader(dataset=test_data, shuffle=True)

# Model definition
model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1)
model = torch.load(
    universel_path + 'Weights/efficientnet7_Class_' + day + '_exp' + str(
        num_exp) + '_' + str(1) + '_par_images.h5')
model.eval()

well_classified = 0
total_samples = 0

# CAM visualization
from pytorch_grad_cam import GradCAM, EigenCAM
from torchvision.models import resnet50

# Initialize CAM
target_layers = [model._blocks[-1]]
cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)

for i, (data, label, img_name) in enumerate(test_loader):
    image = data[0]
    image = image.permute(2, 0, 1)

    data = data.cuda()
    grayscale_cam = cam(input_tensor=data)

    # Perform classification
    output = model(data)
    pred = torch.sigmoid(output)
    predicted_vals = (pred > 0.5).float()


# ...

print(model)
named_layers = dict(model.named_modules())
for l in named_layers:
    print(l)
    print('-------------------')

# ...

# Construct the CAM object once, and then re-use it on many images:
cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)

# ...

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
for i, (data, label, img_name) in enumerate(test_loader):
    image = data[0]
    image = image.permute(2, 0, 1)
    image = image.permute(2, 0, 1)

    data = data.cuda()
    grayscale_cam = cam(input_tensor=data)
    guided_grads = gb_model(data)

    gray_image = cv2.resize(cv2.imread(test_dir + img_name[0], cv2.IMREAD_UNCHANGED), (300, 600)) / 255
    stacked_img = np.stack((gray_image,) * 3, axis=-1)

    output = model(data)

    pred = torch.sigmoid(output)
    predicted_vals = (pred > 0.5).float()
    new_df = pd.DataFrame({'Image_name': [img_name],
                           'label': [label[0].item()], 'prediction_label': [int(predicted_vals)]})
    o_header = False
    print('prediction {} || {} Truth'.format(int(predicted_vals), label[0].item()), img_name)
    if (int(predicted_vals) - label[0]) == 0:
        well_classfied = well_classfied + 1

    sum = sum + 1
    plt.imshow(guided_grads)
    plt.show()
    print(img_name[0], grayscale_cam.shape, np.repeat(grayscale_cam, 3, axis=0).transpose(1, 2, 0).shape)
    cam_gb = guided_grad_cam(np.repeat(grayscale_cam, 3, axis=0).transpose(1, 2, 0), guided_grads)

    cam_gb_norm_orig = normalize_gradient_image(cam_gb)
    cam_gb_norm_orig *= 255
    cam_gb_norm_orig = Image.fromarray(cam_gb_norm_orig.astype("uint8"))
    plt.imshow(cam_gb_norm_orig)
    plt.show()

    tmp = convert_to_grayscale(cam_gb)[0] * gray_image
    plt.imshow(tmp)
    plt.show()
    grayscale_cam_gb_orig = convert_to_grayscale(cam_gb)
    grayscale_cam_gb_orig = np.repeat(grayscale_cam_gb_orig, 3, axis=0)
    grayscale_cam_gb_orig *= 255
    grayscale_cam_gb_orig = Image.fromarray(grayscale_cam_gb_orig.astype("uint8").transpose(1, 2, 0))
    plt.imshow(grayscale_cam_gb_orig)
    plt.show()

    grayscale_cam = grayscale_cam[0, :]

    img_path = test_dir + img_name[0]
    rgb_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 255
    rgb_img = cv2.resize(rgb_img, (250, 500))

    heatmap = hi.gray2cmap(grayscale_cam)
    heatmap = hu.f2l(heatmap)

    plt.imshow(heatmap)
    plt.show()

    gray_image = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), (300, 600)) / 255
    stacked_img = np.stack((gray_image,) * 3, axis=-1)
    img_mask = np.hstack([stacked_img, stacked_img * heatmap])
    plt.imshow(img_mask)
    plt.show()

    # Save the visualization image
    visualization_path = "visualization/{}.png".format(img_name[0])
    cv2.imwrite(visualization_path, img_mask * 255)

    # Print the image name and the path where the visualization is saved
    print("Visualization saved for image:", img_name[0])
    print("Visualization path:", visualization_path)

    # ...

# Print the total number of well-classified images and the overall accuracy
print("Total well-classified images:", well_classified)
print("Overall Accuracy:", well_classified / sum)

