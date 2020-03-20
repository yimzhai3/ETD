#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:55:50 2019

@author: zym
"""

import torch.nn as nn
from torchvision import models
import torch



resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}


#===============================================================================
# 
#===============================================================================
class Feature(nn.Module):
    
    def __init__(self, network_name, network_config):
        super(Feature, self).__init__()
        if network_name == 'resnet':
            self.__resnet_init(network_config.resnet_name)
    
    def __resnet_init(self, resnet_name):
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        
    def forward(self, data):
        feature = self.feature(data)
        return feature.view(feature.size(0), -1)


#===============================================================================
# 
#===============================================================================
def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.05) #bottleneck PADA-0.005;fc PADA-0.01; office-0.05
            m.bias.data.fill_(0)  


class Full_Connect(nn.Module):
    
    def __init__(self, network_name, network_config):
        super(Full_Connect, self).__init__()
        if network_name == 'resnet':
            self.__resnet_init(network_config.fc_in_features,
                               network_config.bottleneck_dim,
                               network_config.class_num,
                               network_config.dropout_p)


    def __resnet_init(self, in_features, bottleneck_dim=256, class_num=31, dropout_p=None):
        if dropout_p == None:
            self.bottleneck = nn.Sequential(
                    nn.Linear(in_features, bottleneck_dim),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                    )
        else:
            self.bottleneck = nn.Sequential(
                    nn.Linear(in_features, bottleneck_dim),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Dropout(p=dropout_p)
                    )
        self.fc = nn.Linear(bottleneck_dim, class_num)                    
        self.full_connect = nn.Sequential(self.bottleneck, self.fc)
            
    def forward(self, data):
        return self.full_connect(data),self.bottleneck(data)

class Full_OT(nn.Module):
    
    def __init__(self, network_name, network_config):
        super(Full_OT, self).__init__()
        if network_name == 'resnet':
            self.__resnet_init(network_config.fc_in_features,
                               network_config.bottleneck_dim,
                               network_config.class_num,
                               network_config.dropout_p)


    def __resnet_init(self, in_features, bottleneck_dim=256, class_num=31, dropout_p=None):
        self.otneck1 = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
        self.otneck2 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
        self.otneck3 = nn.Sequential(
                nn.Linear(512, bottleneck_dim=256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
                    
        self.full_connect = nn.Sequential(self.otneck1, self.otneck2,self.otneck3)
            
    def forward(self, data):
        return self.full_connect(data)
    
class Attention(nn.Module):
    """ attention Layer"""
    def __init__(self, in_dim=31, out_dim=31):
        super(Attention,self).__init__()
        
        self.query_fc = nn.Linear(in_dim, out_dim)
        self.key_fc = nn.Linear(in_dim, out_dim)
#        self.sigm = nn.ReLU()
        self.softmax  = nn.Softmax(dim=-1) 
        
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B * F )
            returns :
                attention: B * B 
        """
#        batchsize, length = x.size()
        proj_query = self.query_fc(x) # B * F
        proj_key  = self.key_fc(y) # B * F
#        energy =  torch.matmul(x, y.transpose(1, 0)) # B * B
#        attention = self.softmax(energy) # B * B
        attention =  torch.matmul(proj_query, proj_key.transpose(1, 0)) # B * B
        attention =  torch.sigmoid(attention)
        
#        attention = self.softmax(attention) # B * B

        return attention


class U(nn.Module):
    def __init__(self,conv_dim = 64):
        super(U,self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(2048,conv_dim),
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                )
        self.fc2 = nn.Linear(conv_dim, 1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class V(nn.Module):
    def __init__(self,conv_dim = 64):
        super(V,self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(2048,conv_dim),
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                )
        self.fc2 = nn.Linear(conv_dim, 1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x