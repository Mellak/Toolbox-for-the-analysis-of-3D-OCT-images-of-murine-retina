import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from random import randrange
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer, required
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
import math

# Set the data directory and other paths
day_data = 'Final_Multi_class_data'
universel_path = 'D:/DL_Project/D0_D2_Classification/' + day_data + '/'

# Custom dataset class
class ImageData(Dataset):
    def __init__(self, df, data_dir, transform, augmentation=None):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.Image_name[index] + '.png'
        label = self.df.label[index]
        one_hot_label = torch.zeros(4)
        one_hot_label[label] = 1

        img_path = os.path.join(self.data_dir, img_name)
        while not os.path.isfile(img_path):
            number = img_name.split("_", -1)[-1]
            number = number.replace('.png', '')
            irand = randrange(0, 512)
            img_name = img_name.replace(number, str(irand))
            img_path = os.path.join(self.data_dir, img_name)

        src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        image = self.transform(image)
        one_hot_label = torch.tensor(one_hot_label, dtype=torch.float)
        return image, one_hot_label, img_name

# Function to compute accuracy
def compute_accuracy(Y_target, hypothesis):
    Y_prediction = hypothesis
    accuracy = ((Y_prediction.data == Y_target.data).float().mean())
    return accuracy.item()

# Set the test transformation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1]),
])

# Set the test directory
test_dir = '/home/youness/data/TX12_D0_png/clean_images/'
epoch    = 10
# Loop over the experiments
for num_exp in range(5):
    # Load the test data
    testdb = pd.read_csv('/home/youness/data/EffNet/Csv_files_m/train_test/' + 'test' + str(num_exp) + '.csv')
    test_data = ImageData(df=testdb, data_dir=test_dir, transform=test_transform)
    test_loader = DataLoader(dataset=test_data, shuffle=True)

    # Load the trained model
    model = torch.load('/home/youness/data/EffNet/Multi_classif/Weights/efficientnet7_4Class_exp' + str(num_exp) + '_'
                       + str(epoch) + '_par_clean_images.h5')
    model.eval()

    # Initialize variables for accuracy calculation and result storage
    correct = 0
    total = 0
    well_classified = 0
    sum_samples = 0

    ocsv_file = '/home/youness/data/EffNet/Multi_classif/Results/' + 'efficientnet7_4Class_exp' + str(num_exp) + '_' \
                + str(epoch) + '_clean_images.csv'
    o_header = True

    obocsv_file = '/home/youness/data/EffNet/Multi_classif/Results_1by1/' + 'efficientnet7_4Class_exp' + str(num_exp) \
                  + '_' + str(epoch) + '_clean_images.csv'
    obo_header = True

    # Iterate over the test data
    for i, (data, label, img_name) in enumerate(test_loader):
        images = data.cuda()
        labels = label.cuda()
        out = model(images)

        total += labels.size(0)
        best_one_arg = torch.argmax(out)
        labels = torch.squeeze(labels)
        gt_arg = torch.argmax(labels)

        correct += compute_accuracy(best_one_arg, gt_arg)

        new_df = pd.DataFrame({'Img_name': [img_name[0]],
                               'label': [gt_arg.cpu().detach().numpy()],
                               'prediction': [best_one_arg.cpu().detach().numpy()]})
        new_df.to_csv(obocsv_file, mode='a', header=obo_header, index=False)
        obo_header = False

        sum_samples += 1

    # Calculate and save the accuracy results
    new_df = pd.DataFrame({'Experience': ['exp' + str(num_exp)],
                           'label': [labels[0].item()],
                           'prediction_label': [round(correct / total, 4)]})
    new_df.to_csv(ocsv_file, mode='a', header=o_header, index=False)
    print("We are in experience" + str(num_exp) + ", Test Accuracy: {}\n".format(round(correct / total, 4)))

