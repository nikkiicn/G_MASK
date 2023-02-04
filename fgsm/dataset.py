from datasets import Image
import torch
import glob
import os
import csv
import cv2 as cv
import numpy as np
import random
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

# 因为图片大小不一，所以要对图片进行transform
transform = transforms.Compose([transforms.RandomHorizontalFlip(), # 随机水平翻转
                                transforms.ToTensor(), # 转成张量
                                transforms.Normalize([0.5], [0.5])]) # 标准化


class MyDataset(Dataset):
    def __init__(self, filename,  transform = None):
        super(MyDataset, self).__init__()
        # 读取图片及其标签
        with open(os.path.join(filename)) as file:
            reader = csv.reader(file)
            images, labels = [], []
            for img, label in reader:
                images.append(img)
                labels.append(int(label))
        self.images, self.labels = images, labels
        self.transform = transform
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, item):
        # 读取图片
        image = Image.open(os.path.join(self.images[item]))
        # 转换
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.array(self.labels[item]))
        return image, label


if __name__ == '_main_':
    filename = '../../data/FaceRecognition/securityAI/securityAI_round1_dev.csv'
    dataset = MyDataset(filename=filename, transform = transform)
    # 返回 batch 的数据对象
    test_loader = DataLoader(dataset, shuffle = False, batch_size = 64)
    # 生成可迭代对象
    image_test, label_test = iter(test_loader).next()
    # 选择第一张图片
    image_test_sample, label_test_sample = image_test[0].squeeze(), label_test[0]
    # 进行轴转化，因为tensor的三通道为(C, H, W)，要转成(H, W, C)
    image_test_sample = image_test_sample.permute((1, 2, 0)).numpy()
    # 因为前面以标准差和均值都是0.5标准化了图片，所以要转回来
    image_test_sample = image_test_sample * 0.5
    image_test_sample = image_test_sample + 0.5
    # 限制像素值的大小
    image_test_sample = np.clip(image_test_sample, 0, 1)
    # 显示
    plt.subplot(121)
    plt.imshow(image_test_sample)
    plt.title(label_test_sample.item())
    plt.axis('off')
    plt.show()