

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
import numpy as np
from dataset import get_data


class UNet_down_block(nn.Module):
    """
        Encoder class
    """
    def __init__(self, input_channel, output_channel, pretrained_weights, pretrained=False):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
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
        x = self.relu(x)
        x = self.relu(self.conv2(x))
        return x


class UNet_up_block(nn.Module):
    """
        Decoder class
    """
    def __init__(self, prev_channel, input_channel, output_channel, pretrained_weights, pretrained=False):
        super(UNet_up_block, self).__init__()
        self.tr_conv_1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

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
        x = self.relu(x)
        x = torch.cat((x, prev_feature_map), dim=1)
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


    def train_net(self, pre_model, lr, batch_size, log_after):
        self.train()
        # define loss and optimizer
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.NLLLoss2d()
        # print(dataloader)
        dataloader = get_data(batch_size=batch_size)
        if True:
            i = 0
            if pre_model:
                self.load_state_dict(torch.load(pre_model))
                print('log: resumed model {} successfully!'.format(pre_model))
            else:
                print('log: starting anew...')
            while True:
                i += 1
                if not os.path.exists('models/model-{}.pt'.format(i)):
                    torch.save(self.state_dict(), 'models/model-{}.pt'.format(i))
                    print('log: saved models/model-{}.pt'.format(i))
                net_loss = []
                net_accuracy = []
                for idx, data in enumerate(dataloader):
                    image, label = data['data'], data['label'].view(-1).long()
                    # print(image.size())
                    test_x = torch.FloatTensor(image).cuda()
                    # forward
                    out_x, _ = self.forward(test_x)
                    # print(label.numpy().min(), label.numpy().max(), np.unique(label.numpy()))
                    out_x = out_x.view(-1, self.out_classes).cpu()
                    pred = torch.argmax(out_x, dim=1).cpu()
                    # print(out_x.size(), label.size())
                    loss = criterion(out_x, label)
                    # get accuracy metric
                    accuracy = (pred == label).sum()*100/(batch_size*512**2)
                    if idx % log_after == 0 and idx > 0:
                        print('size = {}, loss = {}: accuracy = {:.5f}%'.format(image.size(), loss.item(), accuracy))
                    #################################
                    # three steps for backprop
                    self.zero_grad()
                    loss.backward()
                    optimizer.step()
                    net_accuracy.append(accuracy)
                    net_loss.append(loss.item())
                    #################################
                mean_accuracy = np.asarray(net_accuracy).mean()
                mean_loss = np.asarray(net_loss).mean()
                print('####################################')
                print('epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(i, mean_loss, mean_accuracy))
                print('####################################')
        pass

    def eval_net(self):
        self.eval()

        pass


























