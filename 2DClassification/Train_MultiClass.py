import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from random import randrange
from PIL import Image

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# Set the train directory path
train_dir = '/home/youness/data/TX12_D0_png/clean_images/'

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
            number = img_name.split("_", -1)[-1].replace('.png', '')
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
        return image, one_hot_label


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

# Define the EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=4)

def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.
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

# Load the labels from CSV file
num_exp = 0
labels = pd.read_csv('/home/youness/data/EffNet/Csv_files_m/train_test/train{}.csv'.format(num_exp))
train_data = ImageData(df=labels, data_dir=train_dir, transform=train_transform)

mbatch_size = 4
train_loader = DataLoader(dataset=train_data, batch_size=mbatch_size, shuffle=True)

def BCEWithLogitsLoss_one_hot(out, target):
    _, labels = target.max(dim=1)
    criterion = nn.CrossEntropyLoss().cuda()
    return criterion(out, labels)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

model.cuda()
best_acc = -1
start_epoch = 0

if os.path.exists('/home/youness/data/EffNet/Multi_classif/checkpoints'):
    model, optimizer, start_epoch, best_acc = load_checkpoint(
        model,
        optimizer,
        filename='/home/youness/data/EffNet/Multi_classif/checkpoints/efficientnet7_4Class_checkpoint_exp{}_par_clean_images.pth.tar'.format(
            num_exp)
    )
else:
    os.makedirs('/home/youness/data/EffNet/Multi_classif/checkpoints')

from tqdm import trange

for epoch in trange(start_epoch, 2):  # loop over the dataset multiple times
    running_loss = 0.0
    model.train()
    correct = 0
    total = 0
    batch_idx = 0
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()
        out = model(images)
        loss = BCEWithLogitsLoss_one_hot(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        best_one_arg = torch.argmax(out, dim=1)
        labels = torch.squeeze(labels)
        gt_arg = torch.argmax(labels, dim=1)
        correct += compute_accuracy(best_one_arg, gt_arg)
        if (batch_idx % 256 == 0):
            print('     The loss is:', running_loss / mbatch_size, '     batch_idx:', batch_idx, '/',
                  17 * 4 * 512 / mbatch_size, ' accuracy is:', round(correct / total, 4))
        batch_idx = batch_idx + 1

    print("Epoch: {}, Loss: {}, Train Accuracy: {}".format(epoch, running_loss, round(correct / batch_idx, 4)))
    torch.save(
        model,
        '/home/youness/data/EffNet/Multi_classif/Weights/efficientnet7_4Class_exp{}_{}_par_clean_images.h5'.format(
            num_exp, epoch)
    )
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(
        state,
        '/home/youness/data/EffNet/Multi_classif/checkpoints/efficientnet7_4Class_checkpoint_exp{}_par_clean_images.pth.tar'.format(
            num_exp)
    )

    if (best_acc < round(correct / total, 4)):
        best_acc = round(correct / total, 4)
        torch.save(
            model,
            '/home/youness/data/EffNet/Multi_classif/Weights/efficientnet7_4_class_Best_Model_exp{}_par_clean_images.h5'.format(
                num_exp)
        )

