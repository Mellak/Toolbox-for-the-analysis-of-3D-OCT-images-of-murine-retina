# Source :https://www.kaggle.com/bitthal/baseline-pytorch-efficientnet
'''
This code trains an EfficientNet model on a dataset of grayscale images,
applying data augmentation techniques and binary cross-entropy loss to optimize the model's performance.
It saves the best models and checkpoints for future use.
'''
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from random import randrange

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from efficientnet_pytorch import EfficientNet

# Set the directory for training images
train_dir = '/home/youness/data/images/' # or it can be to images containing only retina surface

# Define the dataset class
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

        img_path = self.data_dir + img_name
        # Check if the image file exists, otherwise find a similar image
        while not os.path.isfile(img_path):
            number = img_name.split("_", -1)[-1]
            number = number.replace('.png', '')
            irand = randrange(0, 512)
            img_name = img_name.replace(number, str(irand))
            img_path = self.data_dir + img_name

        # Read and preprocess the image
        src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        image = self.transform(image)

        return image, label

# Define the transformations for training and testing images
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(2),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

universel_path = 'D:/DL_Project/Uveitis21Internship/2DClassification/'
# Set the days and experiment range
days = ['D2', 'D6', 'D14']
for day in days:
    for num_exp in range(5):
        # Create the EfficientNet model
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1)

        # Function to load checkpoint for the model
        def load_checkpoint(model, optimizer, filename):
            start_epoch = 0
            best_acc = -1
            if os.path.isfile(filename):
                print("=> Loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> Loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
                best_acc = checkpoint['best_acc']
            else:
                print("=> No checkpoint found at '{}'".format(filename))

            return model, optimizer, start_epoch, best_acc

        # Read labels from CSV
        labels = pd.read_csv(universel_path + 'Csv_files/train_test_'+day+'/' + 'train'+str(num_exp)+'_'+day+'.csv')

        # Create the training dataset
        train_data = ImageData(df=labels, data_dir=train_dir, transform=train_transform)

        # Set the batch size and create the data loader
        mbatch_size = 4
        train_loader = DataLoader(dataset=train_data, batch_size=mbatch_size, shuffle=True)

        # Define the loss function
        criterion = nn.BCEWithLogitsLoss().cuda()

        # Define a helper function for calculating BCEWithLogitsLoss for one-hot labels
        def BCEWithLogitsLoss_one_hot(out, target):
            _, labels = target.max(dim=1)
            return nn.CrossEntropyLoss()(out, labels).cuda()

        # Define the optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

        # Move the model to GPU
        model.cuda()

        # Initialize variables for tracking best accuracy and start epoch
        best_acc = -1
        start_epoch = 0

        # Function to compute accuracy
        def compute_accuracy(Y_target, hypothesis):
            Y_prediction = hypothesis
            accuracy = ((Y_prediction.data == Y_target.data).float().mean())
            return accuracy.item()

        # Check if checkpoints directory exists, otherwise create it
        if not os.path.exists(universel_path + 'checkpoints/'):
            os.makedirs(universel_path + 'checkpoints/')

        # Training loop
        from tqdm import trange

        for epoch in trange(start_epoch, 2):  # loop over the dataset for a certain number of epochs
            running_loss = 0.0
            model.train()
            correct = 0
            total = 0
            batch_idx = 0
            for images, labels in train_loader:
                images = images.cuda()
                labels = labels.cuda()
                out = model(images)
                labels = labels.unsqueeze(1).float()
                loss = criterion(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                total += labels.size(0)
                out = torch.sigmoid(out)
                correct += ((out > 0.5).int() == labels).sum().item()

                if batch_idx % 256 == 0:
                    print('     The loss is:', running_loss/mbatch_size, '     batch_idx:', batch_idx, '/', 21*512/mbatch_size,
                          ' accuracy is:', round(correct / total, 4))
                batch_idx = batch_idx + 1

            print("Epoch: {}, Loss: {}, Train Accuracy: {}".format(epoch, running_loss, round(correct / batch_idx, 4)))

            # Save the model and checkpoint
            torch.save(model, universel_path + 'Weights/efficientnet7_Class_'+day+'_exp'+str(num_exp)+'_'+str(epoch)+'_par.h5')
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_acc': best_acc}
            torch.save(state, universel_path + 'checkpoints/efficientnet7_checkpoint_'+day+'_exp'+str(num_exp)+'_par.pth.tar')

            # Save the best model based on accuracy
            if best_acc < round(correct / total, 4):
                best_acc = round(correct / total, 4)
                torch.save(model, universel_path + 'Weights/efficientnet7_2_class_Best_Model_' + day + '_exp' + str(num_exp) + '_par.h5')

