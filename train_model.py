

from __future__ import print_function
from __future__ import division
import torch
import os
import pickle
import numpy as np
from model import tiny_UNet, Bigger_Unet, train_net, inference
import argparse

if __name__ == '__main__':

    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', dest='function', default='train_net')
    parser.add_argument('-d', '--dataset', dest='dataset', default=None)
    parser.add_argument('-e', '--encoder', dest='VGGencoder', default=True)
    parser.add_argument('-i', '--image_size', dest='im_size', default=1024)
    parser.add_argument('-p', '--patch_size', dest='patch_size', default=128)
    parser.add_argument('-l', '--lr', dest='lr', default=1e-3)
    parser.add_argument('-log', '--log_after', dest='log_after', default=10)
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=4)
    parser.add_argument('-m', '--pretrained_model', dest='pre_model', default=None)
    parser.add_argument('-c', '--cuda', dest='cuda', default=0)
    args = parser.parse_args()

    function = args.function
    dataset = args.dataset
    pretrained = args.VGGencoder
    im_size = int(args.im_size)
    patch_size = int(args.patch_size)
    batch_size = int(args.batch_size)
    log_after = int(args.log_after)
    lr = float(args.lr)
    pre_model = args.pre_model
    cuda = int(args.cuda)

    # dir_path = '/home/annuszulfiqar/forest_cover/'
    channels = 1
    net = Bigger_Unet(input_channels=channels, output_classes=16) # we have 11 classes, '0' meaning no class

    function_to_call = eval(function)
    function_to_call(model=net, dataset=dataset, pre_model=pre_model, im_size=im_size, patch_size=patch_size,
                     lr=lr, batch_size=batch_size, log_after=log_after, cuda=cuda)










