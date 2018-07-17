

"""
    Trains the model to learn the class difference in PRImA dataset
"""

from __future__ import print_function
from __future__ import division
import torch
import os
import pickle
import numpy as np
from model import UNet
import argparse

if __name__ == '__main__':

    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lr', dest='lr', default=1e-3)
    parser.add_argument('-log', '--log_after', dest='log_after', default=10)
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=4)
    parser.add_argument('-m', '--pretrained_model', dest='pre_model', default=None)
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    log_after = int(args.log_after)
    lr = float(args.lr)
    pre_model = args.pre_model

    dir_path = '/home/annuszulfiqar/forest_cover/'
    channels, patch = 1, 256
    net = UNet(model_dir_path=os.path.join(dir_path, 'Unet_pretrained_model.pkl'),
               input_channels=channels, output_classes=11).cuda() # we have 11 classes, '0' meaning no class
    net.train_net(pre_model=pre_model, lr=lr, batch_size=batch_size, log_after=log_after)










