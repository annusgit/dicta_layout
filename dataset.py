

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
import matplotlib.pyplot as pl

def get_data(batch_size=4):
    dir_path = 'dataset'
    # dir_path='/home/annus/Desktop/logical_layout_analysis/icdar2017/datasetforlayoutanalysis/PRImA/unziped'
    # val = pickle.load(open(os.path.join(dir_path, 'PRImA_dataset.pkl')))
    val = pickle.load(open('1_100.pkl', 'rb'))
    val_data = val['examples']
    val_label = val['labels']

    def image_to_batch(arr):
        count = 0
        X = 512
        batch = []
        for i in range(2):
            for j in range(2):
                count += 1
                batch.append(arr[i*X:i*X+X, j*X:j*X+X])
        return np.asarray(batch)

    # convert images to batches of images by dividing into 4 parts each...
    images, labels = [], []
    for i in range(val_data.shape[0]):
        images.append(image_to_batch(val_data[i]))
        labels.append(image_to_batch(val_label[i]))
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    # print(images.shape, labels.shape)
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
            pass

        def __getitem__(self, item):
            if self.transform:
                return {'data': self.transform(self.data[item]), 'label': self.labels[item]}
            return {'data': self.data[item], 'label': self.labels[item]}

        def __len__(self):
            return self.data.shape[0]

    data = dataset(data=torch.Tensor(np.expand_dims(images, axis=3).transpose(0,3,2,1)), labels=torch.Tensor(labels), transform=None)
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
    get_data()

















