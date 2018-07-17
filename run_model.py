

"""
    Model for Unet (pretrained in Matlab)
"""

from __future__ import print_function
from __future__ import division
import torch
import os
import pickle
import numpy as np
from model import UNet


if __name__ == '__main__':

    dir_path = '/home/annuszulfiqar/forest_cover/'
    batch_size, channels, patch = 1, 3, 256
    net = UNet(model_dir_path=os.path.join(dir_path, 'Unet_pretrained_model.pkl'),
               input_channels=channels, output_classes=9).cuda() # we have 8 classes, '0' meaning no class
    val = pickle.load(open('dataset/PRImA_dataset.pkl'))
    val_data = val['examples']
    val_label = val['labels']
    # val_data = val_data.transpose((0, 2, 3, 1))
    print('before padding: val_data shape = {}, val_label shape = {}'.format(val_data.shape, val_label.shape))
    net_accuracy = []
    zero_count = 0
    veg_count = 0
    if True:
        for i in range(val_data.shape[0]):
            image = val_data[i,:,:,:]
            # print(image.shape)
            label = val_label[i,]
            print(image.shape, label.shape)
            test_x = torch.FloatTensor(image.transpose(2,1,0)).unsqueeze(0).cuda()
            out_x, pred = net(test_x)
            pred = pred.squeeze(0).cpu().numpy().transpose(1,0)+1
            # get accuracy metric
            accuracy = (pred == label).sum() * 100 / 1024**2
            print(': accuracy = {:.5f}%'.format(accuracy))
            net_accuracy.append(accuracy)
        mean_accuracy = np.asarray(net_accuracy).mean()
        print('total accuracy = {:.5f}%'.format(mean_accuracy))





