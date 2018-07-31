

"""
    UNet model definition in here
"""

from __future__ import print_function
from __future__ import division
import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from dataset import get_data
from get_dataloaders import get_dataloaders
import torch.nn.functional as F
# from torchsummary import summary


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


###########################################################################################

class d2_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(d2_conv_block, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, padding=0)
        self.conv2d_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, padding=1)
        self.conv2d_5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=5, padding=2)
        self.conv2d_7 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=7, padding=3)

    def forward(self, x):
        x1 = self.conv2d_1(x)
        x3 = self.conv2d_3(x)
        x5 = self.conv2d_5(x)
        x7 = self.conv2d_7(x)
        return x1, x3, x5, x7


class d1_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(d1_conv_block, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, padding=0)
        self.conv1d_3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=5, padding=2)
        self.conv1d_7 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=7, padding=3)

    def forward(self, x):
        x1 = self.conv1d_1(x)
        x3 = self.conv1d_3(x)
        x5 = self.conv1d_5(x)
        x7 = self.conv1d_7(x)
        return x1, x3, x5, x7


class d1_model(nn.Module):

    def __init__(self, spatial_size):
        super(d1_model, self).__init__()
        self.input_size = spatial_size
        self.out_classes = 5
        self.conv2d_block = d2_conv_block(in_channels=3, out_channels=3)
        self.bn_1 = nn.BatchNorm2d(3)
        self.conv1d_block1 = d1_conv_block(in_channels=3, out_channels=1)
        self.bn_2 = nn.BatchNorm2d(1)
        self.conv1d_block2 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_3 = nn.BatchNorm2d(1)
        self.conv1d_block3 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_4 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(in_features=self.input_size**2/2, out_features=5)
        self.softmax = nn.Softmax(dim=1)
        # self.dropout2d = nn.Dropout2d(0.5)
        # self.dropout1d1 = nn.Dropout(0.5)
        # self.dropout1d2 = nn.Dropout(0.5)
        # count parameters
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pass

    def forward(self, x):
        batch_size = x.size(0)
        # normalize input anyway...
        # x = (x - torch.mean(x)) / torch.std(x)

        # 2d parallel convs
        x1, x3, x5, x7 = self.conv2d_block(x)
        # add 2d conv feature maps
        out1 = x1 + x3 + x5 + x7
        out1 = self.bn_1(out1).view(batch_size, 3, -1)

        # 1d parallel convs
        # print(out1.size())
        x1, x3, x5, x7 = self.conv1d_block1(out1)
        # print(x1.size(), x3.size(), x5.size(), x7.size())
        # add 1d conv feature maps
        out2 = (x1 + x3 + x5 + x7).view(batch_size, 1, self.input_size, self.input_size)
        out2 = self.bn_2(out2).view(batch_size, 1, -1)
        # out2 = self.dropout1d1(out2)

        # 1d parallel convs
        x1, x3, x5, x7 = self.conv1d_block2(out2)
        # add 1d conv feature maps
        out3 = x1 + x3 + x5 + x7
        # add previous connections
        out3 = (out2 + out3).view(batch_size, 1, self.input_size, self.input_size)
        out3 = self.bn_3(out3).view(batch_size, 1, -1)
        # out3 = self.dropout1d2(out3)

        # 1d parallel convs
        x1, x3, x5, x7 = self.conv1d_block3(out3)
        # add 1d conv feature maps
        out4 = x1 + x3 + x5 + x7
        # add previous connections
        out4 = out2 + out3 + out4
        out4 = self.maxpool(out4).view(batch_size,-1)
        # print(out1.size(), out2.size(), out3.size(), out4.size())
        out = self.softmax(self.fc(out4))
        return out, torch.argmax(out, dim=1)
###############################################################################3


class d1_model_with_dropout(nn.Module):

    def __init__(self, spatial_size):
        super(d1_model_with_dropout, self).__init__()
        self.input_size = spatial_size
        self.out_classes = 5

        self.input_bn = nn.BatchNorm2d(3)
        self.conv2d_block = d2_conv_block(in_channels=3, out_channels=3)
        self.bn_1 = nn.BatchNorm2d(3)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)

        self.conv1d_block1 = d1_conv_block(in_channels=3, out_channels=1)
        self.bn_2 = nn.BatchNorm2d(1)

        self.conv1d_block2 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_3 = nn.BatchNorm2d(1)

        self.conv1d_block3 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_4 = nn.BatchNorm2d(1)

        self.conv1d_block4 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_5 = nn.BatchNorm2d(1)

        self.conv1d_block5 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_6 = nn.BatchNorm2d(1)

        self.conv1d_block6 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_7 = nn.BatchNorm2d(1)

        self.conv1d_block7 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_8 = nn.BatchNorm2d(1)

        self.conv1d_block8 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_9 = nn.BatchNorm2d(1)

        self.conv1d_block9 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_10 = nn.BatchNorm2d(1)

        self.conv1d_block10 = d1_conv_block(in_channels=1, out_channels=1)
        self.bn_11 = nn.BatchNorm2d(1)

        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.fc_1 = nn.Linear(in_features=8, out_features=8)
        self.fc_2 = nn.Linear(in_features=8, out_features=8)
        self.fc_3 = nn.Linear(in_features=8, out_features=self.out_classes)
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.dropout_2d = nn.Dropout2d(0.5)
        self.dropout_1d = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.2)

        # count parameters
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self.num_trainable_params)

        pass

    def d1_unit(self, x_in, conv_block, bn_block):
        # 1d parallel convs
        x1, x3, x5, x7 = conv_block(x_in)
        # add 1d conv feature maps
        out = (x1 + x3 + x5 + x7)
        # add prev maps
        out = (out + x_in).unsqueeze(1)
        out = bn_block(out).squeeze(1)
        out = self.maxpool1d(out)
        return out


    def forward(self, x):
        batch_size = x.size(0)
        # normalize input anyway...
        x = self.input_bn(x)

        # 2d parallel convs
        x1, x3, x5, x7 = self.conv2d_block(x)
        # add 2d conv feature maps
        out1 = x1 + x3 + x5 + x7
        out1 = self.bn_1(out1)
        out1 = self.maxpool2d(out1)
        out1 = self.dropout_2d(out1).view(batch_size, 3, -1) # reshape for 1d ops

        # 1d parallel convs
        x1, x3, x5, x7 = self.conv1d_block1(out1)
        # add 1d conv feature maps
        out2 = (x1 + x3 + x5 + x7).unsqueeze(1)
        out2 = self.bn_2(out2).squeeze(1)
        out2 = self.maxpool1d(out2)

        out3 = self.d1_unit(x_in=out2, conv_block=self.conv1d_block2, bn_block=self.bn_3)
        out4 = self.d1_unit(x_in=out3, conv_block=self.conv1d_block3, bn_block=self.bn_4)
        out5 = self.d1_unit(x_in=out4, conv_block=self.conv1d_block4, bn_block=self.bn_5)
        out5 = self.dropout_1d(out5)
        out6 = self.d1_unit(x_in=out5, conv_block=self.conv1d_block5, bn_block=self.bn_6)
        out7 = self.d1_unit(x_in=out6, conv_block=self.conv1d_block6, bn_block=self.bn_7)
        out8 = self.d1_unit(x_in=out7, conv_block=self.conv1d_block7, bn_block=self.bn_8)
        out9 = self.d1_unit(x_in=out8, conv_block=self.conv1d_block8, bn_block=self.bn_9)
        out10 = self.d1_unit(x_in=out9, conv_block=self.conv1d_block9, bn_block=self.bn_10)
        out10 = out10.squeeze(dim=1)

        out = self.relu(self.fc_1(out10))
        out = self.dropout_fc(out)
        out = self.relu(self.fc_2(out))
        out = self.softmax(self.fc_3(out))
        return out, torch.argmax(out, dim=1)
###############################################################################3


class d1_model_(nn.Module):

    def __init__(self, spatial_size):
        super(d1_model_, self).__init__()
        self.input_size, self.out_classes = spatial_size, 5
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=2)
        self.conv2d_3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, padding=3)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_6 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_7 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_8 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_9 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(in_features=self.input_size**2/(2**10), out_features=5)
        self.softmax = nn.Softmax(dim=1)
        # self.dropout2d = nn.Dropout2d(0.5)
        # self.dropout1d1 = nn.Dropout(0.5)
        # self.dropout1d2 = nn.Dropout(0.5)

        # count parameters
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pass

    def forward(self, x):
        batch_size = x.size(0)
        # normalize input anyway...
        c2_1 = self.conv2d_1(x) # image_size*image_size
        c2_2 = self.conv2d_2(x)  # image_size*image_size
        c2_3 = self.conv2d_3(x)  # image_size*image_size
        out1_ = c2_1 + c2_2 + c2_3
        out1 = self.maxpool2d(out1_).view(batch_size,1,-1) # image_size/2*image_size/2
        out2_ = self.conv1d_1(out1) + out1 # image_size/2*image_size/2
        out2 = self.maxpool1d(out2_)
        out3_ = self.conv1d_2(out2) + out2
        out3 = self.maxpool1d(out3_)
        out4_ = self.conv1d_3(out3) + out3
        out4 = self.maxpool1d(out4_)
        out5_ = self.conv1d_4(out4) + out4
        out5 = self.maxpool1d(out5_)
        out6_ = self.conv1d_5(out5) + out5
        out6 = self.maxpool1d(out6_)
        out7_ = self.conv1d_6(out6) + out6
        out7 = self.maxpool1d(out7_)
        out8_ = self.conv1d_7(out7) + out7
        out8 = self.maxpool1d(out8_)
        out9_ = self.conv1d_8(out8) + out8
        out9 = self.maxpool1d(out9_)
        out = self.softmax(self.fc(out9).squeeze())
        # count parameters
        # self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print('log: num of trainable params: ', self.num_trainable_params)
        return out, torch.argmax(out, dim=1)
###############################################################################3


class d1_resnet(nn.Module):

    def __init__(self, spatial_size):
        super(d1_resnet, self).__init__()

        self.input_size, self.out_classes = spatial_size, 5
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=2)
        self.conv2d_3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, padding=3)

        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_6 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_7 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_8 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_9 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_10 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.input_size**2/(2**7), out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=5)
        self.softmax = nn.Softmax(dim=1)

        self.dropout2d = nn.Dropout2d(0.5)
        self.dropout1d1 = nn.Dropout(0.2)
        self.dropoutfc = nn.Dropout(0.2)

        # count parameters
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('log: trainable params', self.num_trainable_params)
        pass

    def residual_module(self, x_in, conv_1, conv_2):
        # implements a single res-block
        # x -> weight1 -> relu -> weight2 -> F(x)
        # F(x) + x -> relu -> output
        return self.relu(conv_2(self.relu(conv_1(x_in))) + x_in)

    def forward(self, x):
        batch_size = x.size(0)
        # normalize input anyway...
        c2_1 = self.conv2d_1(x) # image_size*image_size
        c2_2 = self.conv2d_2(x)  # image_size*image_size
        c2_3 = self.conv2d_3(x)  # image_size*image_size
        out1_ = self.relu(c2_1 + c2_2 + c2_3)
        out1 = self.maxpool2d(out1_).view(batch_size,1,-1) # image_size/2*image_size/2
        out1 = self.dropout2d(out1)

        # out2 = self.relu(self.conv1d_1(out1)) # + out1 # image_size/2*image_size/2
        # out3 = self.relu(self.conv1d_2(out2) + out1)
        out3 = self.residual_module(x_in=out1, conv_1=self.conv1d_1, conv_2=self.conv1d_2)
        out3 = self.maxpool1d(out3)

        # out4 = self.relu(self.conv1d_3(out3))
        # out5 = self.relu(self.conv1d_4(out4) + out3)
        out5 = self.residual_module(x_in=out3, conv_1=self.conv1d_3, conv_2=self.conv1d_4)
        out5 = self.maxpool1d(out5)

        # out6 = self.relu(self.conv1d_5(out5))
        # out7 = self.relu(self.conv1d_6(out6) + out5)
        out7 = self.residual_module(x_in=out5, conv_1=self.conv1d_5, conv_2=self.conv1d_6)
        out7 = self.maxpool1d(out7)
        out7 = self.dropout1d1(out7)

        # out8 = self.relu(self.conv1d_7(out7))
        # out9 = self.relu(self.conv1d_8(out8) + out7)
        out9 = self.residual_module(x_in=out7, conv_1=self.conv1d_7, conv_2=self.conv1d_8)
        out9 = self.maxpool1d(out9)

        # out10 = self.relu(self.conv1d_9(out9))
        # out11 = self.relu(self.conv1d_10(out10) + out9)
        out11 = self.residual_module(x_in=out9, conv_1=self.conv1d_9, conv_2=self.conv1d_10)
        out11 = self.maxpool1d(out11)

        out = self.relu(self.fc1(out11).squeeze())
        out = self.dropoutfc(out)
        out = self.relu(self.fc2(out))
        out = self.softmax(self.fc3(out))
        # count parameters
        # self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print('log: num of trainable params: ', self.num_trainable_params)
        return out, torch.argmax(out, dim=1)
###############################################################################3


class LayNet(nn.Module):

    def __init__(self):
        super(LayNet, self).__init__()

        pass

    def forward(self, x):
        # x will be a tuple of as many receptive fields of 8*8 as we want


        pass



def train_net(model, dataset, pre_model, model_dir, im_size, patch_size, lr, batch_size, log_after, cuda):
    model.train()
    print(model)
    if cuda:
        print('GPU')
        model.cuda()
    # define loss and optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_dataloader, test_loader = get_dataloaders(base_path=dataset, batch_size=batch_size)
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
            for idx, data in enumerate(train_loader):
                test_x, label = data['input'], data['label']
                if cuda:
                    test_x = test_x.cuda()
                # forward
                out_x, pred = model.forward(test_x)
                size = out_x.shape
                loss = criterion(out_x.cpu(), label)
                # get accuracy metric
                accuracy = (pred.cpu() == label).sum()
                if idx % log_after == 0 and idx > 0:
                    print('image size = {}, out_x size = {}, loss = {}: accuracy = {}/{}'.format(test_x.size(),
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

            # validate model
            if i % 10 == 0:
                eval_net(model=model, criterion=criterion, val_loader=val_dataloader, cuda=cuda)
    pass


def eval_net(model, criterion, val_loader, cuda):
    model.eval()
    net_accuracy, net_loss = [], []
    for idx, data in enumerate(val_loader):
        test_x, label = data['input'], data['label']
        if cuda:
            test_x = test_x.cuda()
        # forward
        out_x, pred = model.forward(test_x)
        loss = criterion(out_x.cpu(), label)
        # get accuracy metric
        accuracy = (pred.cpu() == label).sum()
        accuracy = accuracy * 100 / pred.size(0)
        net_accuracy.append(accuracy)
        net_loss.append(loss.item())
        #################################
    mean_accuracy = np.asarray(net_accuracy).mean()
    mean_loss = np.asarray(net_loss).mean()
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('log: validation:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
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



def test():
    d1_net = d1_resnet(spatial_size=128)
    testing = torch.randn((2,3,128,128))
    out, pred = d1_net(testing)
    print(out.size(), pred.size())
#
test()



















