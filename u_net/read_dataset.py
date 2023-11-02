# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt


class SegmentationDataset(object):
    def __init__(self, image_dir, mask_dir):
        self.images = []
        self.masks = []
        files = os.listdir(image_dir)
        sfiles = os.listdir(mask_dir)
        for i in range(len(sfiles)):
            img_file = os.path.join(image_dir, files[i])
            mask_file = os.path.join(mask_dir, sfiles[i])
            self.images.append(img_file)
            self.masks.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # BGR order
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        # print(img.shape)
        # 输入图像
        img = np.float32(img) / 255.0
        img = np.expand_dims(img, 0)

        # 目标标签0 ~ 1， 对于
        mask[mask <= 128] = 0
        mask[mask > 128] = 1
        mask = np.expand_dims(mask, 0)
        sample = {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), }
        return sample

def imshow_image(mydata_loader):
    plt.figure()
    for (cnt, i) in enumerate(mydata_loader):
        image = i['image']
        label = i['mask']

        for j in range(8):  # 一个批次设为：8
            # ax = plt.subplot(2, 4, j + 1)
            # ax.axis('off')
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            # permute函数：可以同时多次交换tensor的维度
            # print(image[j].permute(1, 2, 0).shape)
            ax1.imshow(image[j].permute(1, 2, 0), cmap='gray')
            ax1.set_title('image')

            ax2.imshow(label[j].permute(1, 2, 0), cmap='gray')
            ax2.set_title('mask')
            # plt.pause(0.005)
            plt.show()
        if cnt == 6:
            break

    plt.pause(0.005)


if __name__ == '__main__':
    image_dir = 'D:/PycharmProjects/COMP9517-Surfing-Project/u_net/part_images'
    mask_dir = 'D:/PycharmProjects/COMP9517-Surfing-Project/u_net/dask_images/'
    dataloader = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)
    mydata_loader = DataLoader(dataloader, batch_size=8, shuffle=False)
    imshow_image(mydata_loader)
