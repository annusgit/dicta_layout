

"""
    UNet model definition in here
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import os
# from sklearn.preprocessing import scale
import numpy as np
from dataset import get_data
import cv2

class UNet_down_block(nn.Module):
    """
        Encoder class
    """
    def __init__(self, input_channel, output_channel, pretrained_weights=None, pretrained=False):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channel)
        self.bn_2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

        if pretrained:
            # load previously trained weights
            # pretrained_weights = [conv1_W, conv1_b, conv2_W, conv2_b]
            self.conv1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[0].transpose(3, 2, 1, 0)))
            self.conv1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[1].flatten()))
            self.conv2.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[2].transpose(3, 2, 1, 0)))
            self.conv2.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[3].flatten()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn_1(x))
        x = self.relu(self.bn_2(self.conv2(x)))
        return x


class UNet_up_block(nn.Module):
    """
        Decoder class
    """
    def __init__(self, prev_channel, input_channel, output_channel, pretrained_weights=None, pretrained=False):
        super(UNet_up_block, self).__init__()
        self.tr_conv_1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(prev_channel+output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm2d(output_channel)
        self.bn_2 = nn.BatchNorm2d(output_channel)
        self.bn_3 = nn.BatchNorm2d(output_channel)

        if pretrained:
            # load pretrained weights
            # pretrained_weights = [tconv_W, tconv_b, conv1_W, conv1_b, conv2_W, conv2_b]
            self.tr_conv_1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[0].transpose(3, 2, 1, 0)))
            self.tr_conv_1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[1]).view(-1))
            self.conv_1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[2].transpose(3, 2, 1, 0)))
            self.conv_1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[3]).view(-1))
            self.conv_2.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[4].transpose(3, 2, 1, 0)))
            self.conv_2.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[5]).view(-1))

    def forward(self, prev_feature_map, x):
        x = self.tr_conv_1(x)
        # print(prev_feature_map.size(), x.size())
        # x = self.bn_1(x)
        x = self.relu(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        # x = self.relu(self.bn_2(self.conv_1(x)))
        # x = self.relu(self.bn_3(self.conv_2(x)))
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        return x


class UNet(nn.Module):

    def __init__(self, model_dir_path, input_channels, output_classes, pretrained=False):
        self.out_classes = output_classes
        super(UNet, self).__init__()
        # start by loading the pretrained weights from model_dict saved earlier
        import pickle
        with open(model_dir_path, 'rb') as handle:
            model_dict = pickle.load(handle)
            print('log: loaded saved model dictionary')
        print('total number of weights to be loaded into pytorch model =', len(model_dict.keys()))

        # create encoders and pass pretrained weights...
        self.encoder_1 = UNet_down_block(input_channels, 64, [model_dict['e1c1'], model_dict['e1c1_b'],
                                                              model_dict['e1c2'], model_dict['e1c2_b']],
                                         pretrained=False)
        self.encoder_2 = UNet_down_block(64, 128, [model_dict['e2c1'], model_dict['e2c1_b'],
                                                   model_dict['e2c2'], model_dict['e2c2_b']])
        self.encoder_3 = UNet_down_block(128, 256, [model_dict['e3c1'], model_dict['e3c1_b'],
                                                    model_dict['e3c2'], model_dict['e3c2_b']])
        self.encoder_4 = UNet_down_block(256, 512, [model_dict['e4c1'], model_dict['e4c1_b'],
                                                    model_dict['e4c2'], model_dict['e4c2_b']])
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)

        self.mid_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

        if pretrained:
            # load mid_conv weights
            self.mid_conv1.weight = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv1'].transpose(3, 2, 1, 0)))
            self.mid_conv1.bias = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv1_b'][:,:,:].flatten()))
            self.mid_conv2.weight = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv2'].transpose(3, 2, 1, 0)))
            self.mid_conv2.bias = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv2_b']).view(-1))

        # create decoders and load pretrained weights
        self.decoder_1 = UNet_up_block(512, 1024, 512, [model_dict['d1tc'], model_dict['d1tc_b'],
                                                        model_dict['d1c1'], model_dict['d1c1_b'],
                                                        model_dict['d1c2'], model_dict['d1c2_b']])
        self.decoder_2 = UNet_up_block(256, 512, 256, [model_dict['d2tc'], model_dict['d2tc_b'],
                                                       model_dict['d2c1'], model_dict['d2c1_b'],
                                                       model_dict['d2c2'], model_dict['d2c2_b']])
        self.decoder_3 = UNet_up_block(128, 256, 128, [model_dict['d3tc'], model_dict['d3tc_b'],
                                                       model_dict['d3c1'], model_dict['d3c1_b'],
                                                       model_dict['d3c2'], model_dict['d3c2_b']])
        self.decoder_4 = UNet_up_block(64, 128, 64, [model_dict['d4tc'], model_dict['d4tc_b'],
                                                     model_dict['d4c1'], model_dict['d4c1_b'],
                                                     model_dict['d4c2'], model_dict['d4c2_b']])

        self.last_conv = nn.Conv2d(64, self.out_classes, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)

        # if pretrained:
        #     load final_conv weights
        #     pass # not these!!!
        #     self.last_conv.weight = torch.nn.Parameter(torch.Tensor(model_dict['final_conv'].transpose(3, 2, 1, 0)))
        #     self.last_conv.bias = torch.nn.Parameter(torch.Tensor(model_dict['final_conv_b']).view(-1))
        pass

    def forward(self, x):
        self.x1_cat = self.encoder_1(x)
        self.x1 = self.max_pool(self.x1_cat)
        self.x2_cat = self.encoder_2(self.x1)
        self.x2 = self.max_pool(self.x2_cat)
        self.x3_cat = self.encoder_3(self.x2)
        self.x3 = self.max_pool(self.x3_cat)
        self.x4_cat = self.encoder_4(self.x3)
        self.x4_cat_1 = self.dropout(self.x4_cat)
        self.x4 = self.max_pool(self.x4_cat_1)
        self.x_mid = self.mid_conv1(self.x4)
        self.x_mid = self.relu(self.x_mid)
        self.x_mid = self.mid_conv2(self.x_mid)
        self.x_mid = self.relu(self.x_mid)
        self.x_mid = self.dropout(self.x_mid)
        x = self.decoder_1(self.x4_cat_1, self.x_mid)
        x = self.decoder_2(self.x3_cat, x)
        x = self.decoder_3(self.x2_cat, x)
        x = self.decoder_4(self.x1_cat, x)
        x = self.last_conv(x)
        x = self.softmax(x)
        pred = torch.argmax(x, dim=1)
        return x, pred

    def eval_net(self):
        self.eval()

        pass


class tiny_UNet(nn.Module):

    def __init__(self, input_channels, output_classes):
        self.out_classes = output_classes
        super(tiny_UNet, self).__init__()

        # create encoders (input size -> k * k * 3)
        self.encoder_1 = UNet_down_block(input_channels, 64) # (size -> k * k * 64)
        self.max_pool = nn.MaxPool2d(2, 2) # (size -> k/2 * k/2 * 64)
        self.dropout = nn.Dropout2d(0.5)

        self.mid_conv1 = nn.Conv2d(64, 64, 3, padding=1) # (size -> k/2 * k/2 * 64)
        self.mid_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

        # create decoders
        self.decoder_1 = UNet_up_block(64, 64, 64) # (size -> k * k * 64)

        self.last_conv = nn.Conv2d(64, self.out_classes, kernel_size=1) # (size -> k * k * out_classes)
        self.softmax = nn.LogSoftmax(dim=1) # (size -> k * k * 1)
        pass

    def forward(self, x):
        # (input size -> k * k * 3)
        self.x1_cat = self.encoder_1(x) # (size -> k * k * 64)
        self.x1 = self.dropout(self.max_pool(self.x1_cat)) # (size -> k/2 * k/2 * 64)
        self.x_mid = self.mid_conv1(self.x1) # (size -> k/2 * k/2 * 64)
        self.x_mid = self.relu(self.mid_bn(self.x_mid))# (size -> k/2 * k/2 * 64)
        self.x_mid = self.dropout(self.x_mid)# (size -> k/2 * k/2 * 64)
        x = self.decoder_1(self.x1_cat, self.x_mid)# (size -> k * k * 128)
        x = self.last_conv(x)# (size -> k * k * num_classes)
        x = self.softmax(x)# (size -> k * k * num_classes)
        pred = torch.argmax(x, dim=1)# (size -> k * k * 1)
        return x, pred

    def eval_net(self):
        self.eval()

        pass


class Bigger_Unet(nn.Module):

    def __init__(self, input_channels, output_classes):
        self.out_classes = output_classes
        super(Bigger_Unet, self).__init__()
        const_channels = 64

        # create encoders (input size -> k * k * 3)
        self.encoder_1 = UNet_down_block(input_channels, const_channels) # (size -> k * k * 64)
        self.encoder_2 = UNet_down_block(const_channels, const_channels) # (size -> k * k * 64)
        self.max_pool = nn.MaxPool2d(2, 2) # (size -> k/2 * k/2 * 64)
        self.dropout = nn.Dropout2d(0.5)

        self.mid_conv1 = nn.Conv2d(64, 64, 3, padding=1) # (size -> k/2 * k/2 * 64)
        self.mid_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

        # create decoders
        self.decoder_1 = UNet_up_block(const_channels, const_channels, const_channels) # (size -> k * k * 64)
        self.decoder_2 = UNet_up_block(const_channels, const_channels, const_channels) # (size -> k * k * 64)

        self.last_conv = nn.Conv2d(const_channels, self.out_classes, kernel_size=1) # (size -> k * k * out_classes)
        self.softmax = nn.LogSoftmax(dim=1) # (size -> k * k * 1)
        pass

    def forward(self, x):
        # (input size -> k * k * 3)
        self.x1_cat = self.encoder_1(x) # (size -> k * k * 64)
        self.x1 = self.max_pool(self.x1_cat) # (size -> k/2 * k/2 * 64)
        self.x2_cat = self.encoder_2(self.x1) # (size -> k/2 * k/2 * 64)
        self.x2 = self.max_pool(self.x2_cat) # (size -> k/4 * k/4 * 64)
        # dropout some activations
        self.x2 = self.dropout(self.x2)
        self.x_mid = self.mid_conv1(self.x2) # (size -> k/4 * k/4 * 64)
        self.x_mid = self.relu(self.mid_bn(self.x_mid))# (size -> k/4 * k/4 * 64)
        # dropout some activations
        self.x_mid = self.dropout(self.x_mid)# (size -> k/2 * k/2 * 64)
        x = self.decoder_1(self.x2_cat, self.x_mid)# (size -> k * k * 128)
        x = self.decoder_2(self.x1_cat, x)# (size -> k * k * 128)
        x = self.last_conv(x)# (size -> k * k * num_classes)
        x = self.softmax(x)# (size -> k * k * num_classes)
        pred = torch.argmax(x, dim=1)# (size -> k * k * 1)
        return x, pred

    def eval_net(self):
        self.eval()

        pass


def train_net(model, dataset, pre_model, im_size, patch_size, lr, batch_size, log_after, cuda):
    model.train()
    print(model)
    if cuda:
        print('GPU')
        model.cuda()
    # define loss and optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # print(dataloader)
    dataloader = get_data(datasetpkl=dataset, im_size=im_size, patch_size=patch_size, batch_size=batch_size)
    model_dir = 'bigger_models'
    try:
        os.mkdir(model_dir)
    except:
        pass
    if True:
        i = 0
        if pre_model:
            # self.load_state_dict(torch.load(pre_model)['model'])
            model.load_state_dict(torch.load(pre_model))
            print('log: resumed model {} successfully!'.format(pre_model))
        else:
            print('log: starting anew...')
        while True:
            i += 1
            if pre_model:
                model_number = int(pre_model.split('.')[0].split('-')[1])
                save_path = os.path.join(model_dir, 'model-{}.pt'.format(model_number+i))
            else:
                save_path = os.path.join(model_dir, 'model-{}.pt'.format(i))
            if i > 1 and not os.path.exists(save_path):
                torch.save(model.state_dict(), save_path)
                print('log: saved {}'.format(save_path))
            net_loss = []
            net_accuracy = []
            for idx, data in enumerate(dataloader):
                image, label = data['data']/255, data['label'].view(-1).long()
                # print(image.size())
                # np.save('testimage.npy', image[3].numpy().reshape(patch_size, patch_size))
                # np.save('testlabel.npy', label[3].numpy().reshape(patch_size, patch_size))
                # print(image.size())
                test_x = torch.FloatTensor(image)
                if cuda:
                    test_x = test_x.cuda()
                # forward
                out_x, pred = model.forward(test_x)
                size = out_x.shape
                # print(out_x.shape, label.shape)
                # print(label.numpy().min(), label.numpy().max(), np.unique(label.numpy()))
                out_x = out_x.view(-1, model.out_classes).cpu()
                loss = criterion(out_x.cpu(), label)
                # get accuracy metric
                pred = pred.view(-1).cpu()
                # print(pred.size(), label.size())
                accuracy = (pred == label).sum()
                if idx % log_after == 0 and idx > 0:
                    print('image size = {}, out_x size = {}, loss = {}: accuracy = {}/{}'.format(image.size(),
                                                                                                 size,
                                                                                                 loss.item(),
                                                                                                 accuracy,
                                                                                                 pred.size(0)))
                #################################
                # three steps for backprop
                model.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy = accuracy * 100 / pred.size(0)
                net_accuracy.append(accuracy)
                net_loss.append(loss.item())
                #################################
            mean_accuracy = np.asarray(net_accuracy).mean()
            mean_loss = np.asarray(net_loss).mean()
            print('####################################')
            print('epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(i, mean_loss, mean_accuracy))
            print('####################################')
    pass


def inference(model, dataset, pre_model, im_size, patch_size, lr, batch_size, log_after, cuda):
    model.eval()
    if cuda:
        model.cuda()
    # save_path = 'saved'
    dataloader = get_data(datasetpkl=dataset, im_size=im_size, patch_size=patch_size, batch_size=batch_size)
    if pre_model:
        # self.load_state_dict(torch.load(pre_model)['model'])
        if not cuda:
            model.load_state_dict(torch.load(pre_model, map_location=lambda storage, location: storage))
        else:
            model.load_state_dict(torch.load(pre_model))
        print('log: resumed model {} successfully!'.format(pre_model))
    else:
        print('log: starting anew...')
    for idx, data in enumerate(dataloader):
        image, label = data['data'], data['label']
        # import matplotlib.pyplot as pl
        # pl.subplot(121)
        # print(image.size())
        # pl.imshow(image[3].numpy().reshape(patch_size, patch_size))
        # pl.subplot(122)
        # pl.imshow(label[3].numpy().reshape(patch_size, patch_size))
        # pl.show()
        if cuda:
            test_x = torch.FloatTensor(image).cuda()
        else:
            test_x = torch.FloatTensor(image)
        # forward
        _, pred = model.forward(test_x)
        # print(image.size(), pred.size(), label.size())
        # cv2.imwrite('saved/image-{}.jpg'.format(idx), image.cpu().numpy().reshape(patch_size, patch_size))
        # cv2.imwrite('saved/label-{}.jpg'.format(idx), label.cpu().numpy().reshape(patch_size, patch_size))
        # cv2.imwrite('saved/pred-{}.jpg'.format(idx), pred.cpu().numpy().reshape(patch_size, patch_size))
        # np.save('saved/label-{}.npy'.format(idx), label.cpu().numpy())
        # np.save('saved/pred-{}.npy'.format(idx), pred.cpu().numpy())
        if idx % log_after == 0 and idx > 0:
            print('{} {}'.format(pred.size(), image.size()))
    pass





















