

"""
    Creates the dataset
"""

from __future__ import print_function
from __future__ import division
import torch
import os
import time
import pickle
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as pl


def get_data(datasetpkl, im_size, patch_size, batch_size):
    dir_path = 'dataset'
    # dir_path='/home/annus/Desktop/logical_layout_analysis/icdar2017/datasetforlayoutanalysis/PRImA/unziped'
    # val = pickle.load(open(os.path.join(dir_path, 'PRImA_dataset.pkl')))
    print('dataset -> {}'.format(datasetpkl))
    val = pickle.load(open(datasetpkl, 'rb'))
    val_data = val['examples']
    val_label = val['labels']
    print(val_data.shape, val_label.shape)

    def image_to_batch(arr, Img_size, Patch_size):
        count = 0
        batch = []
        parts = Img_size // Patch_size
        for i in range(parts):
            for j in range(parts):
                count += 1
                batch.append(arr[i*Patch_size:i*Patch_size+Patch_size, j*Patch_size:j*Patch_size+Patch_size])
        return np.asarray(batch)

    # convert images to batches of images by dividing into 4 parts each...
    images, labels = [], []
    for i in range(val_data.shape[0]):
        # print(image_to_batch(val_data[i], Img_size=im_size, Patch_size=patch_size).shape)
        images.append(image_to_batch(val_data[i], Img_size=im_size, Patch_size=patch_size))
        labels.append(image_to_batch(val_label[i], Img_size=im_size, Patch_size=patch_size))
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(images.shape, labels.shape)
    # print(images.mean(axis=(0,1,2)), images.std(axis=(0,1,2)))
    # rand = np.random.randint(0,216)
    # image, label = Image.fromarray(images[rand,:,:,:].astype(np.uint8).reshape(512, 512, 3), 'RGB'), labels[rand,:,:].reshape(512, 512)
    # pl.subplot(221)
    # pl.imshow(image)
    # pl.subplot(222)
    # pl.imshow(label)
    # pl.show()
    # time.sleep(100)

    class dataset(Dataset):
        def __init__(self, data, labels, transform=None):
            super(dataset, self).__init__()
            self.data = data
            self.labels = labels
            self.transform = transform
            if self.transform:
                print('We have a transform now!')
            pass

        def __getitem__(self, item):
            if self.transform:
                return {'data': self.transform(self.data[item]), 'label': self.labels[item]}
            return {'data': self.data[item], 'label': self.labels[item]}

        def __len__(self):
            return self.data.shape[0]

    # print(images.shape)
    images = np.expand_dims(images, axis=3)
    # mean = images.mean()
    # std = images.std()
    # print(images.shape)
    # print(mean, std)
    data = dataset(data=torch.Tensor(images.transpose(0,3,1,2)), labels=torch.Tensor(labels))
                   # transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)]))
                   # transform=transforms.Compose([transforms.Normalize(mean=[201.704639206], std=[57.6908601402])]))
    # transform = transforms.Compose([transforms.Normalize(mean=list(mean), std=list(std)])
    dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)
    # for idx, batch in enumerate(dataloader):
    #     print('{} -> '.format(idx+1), batch['data'].shape, batch['label'].shape,
    #           map(type, (batch['data'], batch['label'])))
    #     image, label = Image.fromarray(batch['data'].numpy().astype(np.uint8).transpose(0,3,2,1).reshape(batch_size, 512,512,3),
    #                                    'RGB'), batch['label'].reshape(batch_size, 512, 512)
    #     pl.subplot(221)
    #     pl.imshow(image[2])
    #     pl.subplot(222)
    #     pl.imshow(label[2])
    #     pl.show()
    #     time.sleep(100)

    return dataloader


if __name__ == '__main__':
    get_data(datasetpkl='/home/annus/Desktop/logical_layout_analysis/newbiggerdataset/new_grayscale/fixedagain_gray_scale.pkl',
             im_size=512, patch_size=512, batch_size=4)

















