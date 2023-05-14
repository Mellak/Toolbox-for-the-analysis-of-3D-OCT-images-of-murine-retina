# This code performs image classification using the EfficientNet model trained on grayscale images.
# It loads the test dataset, applies transformations to the images, and evaluates the model's performance.
# The results are saved in a CSV file.

import numpy as np
import pandas as pd
import os
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from random import randrange, uniform
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

# Dataset class for loading images and labels
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

        # Check if the image file exists, if not, replace the image name with a random number until a file is found
        while not os.path.isfile(img_path):
            number = img_name.split("_", -1)[-1]
            number = number.replace('.png', '')
            irand = randrange(0, 512)
            img_name = img_name.replace(number, str(irand))
            img_path = self.data_dir + img_name

        src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        image = self.transform(image)

        return image, label, img_name


# Define the image transformations for testing
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1]),
])

# Set the test directory and day
universel_path = 'D:/DL_Project/Uveitis21Internship/2DClassification/'
test_dir = '/home/youness/data/TX12_D0_png/clean_images/'
day = 'D2' # Can be D2, D6, D14
epoch    = 10

# Iterate over different experiments
for num_exp in range(5):
    # Load the test dataset CSV file
    testdb = pd.read_csv(universel_path + 'Csv_files/train_test_'+day+'/' + 'test'+str(num_exp)+'_'+day+'.csv')

    # Create the test dataset
    test_data = ImageData(df=testdb, data_dir=test_dir, transform=test_transform)
    test_loader = DataLoader(dataset=test_data, shuffle=False)

    # Load the pre-trained model
    model = torch.load(universel_path + 'Weights/efficientnet7_Class_'+day+'_exp'+str(num_exp)+'_'+str(epoch)+'_par_clean_images.h5')
    model.eval()

    correct = 0
    total = 0
    ocsv_file = universel_path + 'Results_1by1/' + 'efficientnet7_Class_'+day+'_exp'+str(num_exp)+'_'+str(epoch)+'_clean_images.csv'
    o_header = True

    # Evaluate the model on the test dataset
    for i, (data, label, img_name) in enumerate(test_loader):
        images = data.cuda()
        labels = labels.cuda()

        # Perform forward pass
        out = model(images)
        labels = labels.unsqueeze(1).float()

        total += labels.size(0)
        out = torch.sigmoid(out)

        # Count the number of correct predictions
        correct += ((out > 0.5).int() == labels).sum().item()

        # Save the prediction results to a CSV file
        new_df = pd.DataFrame({'Img_name': [img_name[0]],
                               'label': [label[0].item()], 'prediction': [(out > 0.5).int().item()]})
        new_df.to_csv(ocsv_file, mode='a', header=o_header, index=False)
        o_header = False

        # Calculate the test accuracy
    test_accuracy = round(correct / total, 4)
    print("Experience {}: Test Accuracy: {}".format(num_exp, test_accuracy))

# End of the code

