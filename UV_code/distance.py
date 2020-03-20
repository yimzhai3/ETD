# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:40:56 2019

@author: de'l'l
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:29:15 2019

@author: 11651
"""

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils import model_zoo
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# Training settings
#torch.manual_seed(1) 

epochs = 50
n_classes = 31
train_batch_size = 32
g_conv_dim = 64
lamda = 1

source_domain = 'webcam'
target_domain = 'amazon'
##################################
# Load data and create Data loaders
##################################
    
class U(nn.Module):
    def __init__(self, conv_dim=64):
        super(U, self).__init__()
        self.fc1 = nn.Sequential( 
              nn.Linear(2048, conv_dim), 
              nn.LeakyReLU(negative_slope=0.2, inplace=True), 
              #nn.Dropout(p=0.2) 
             ) 
        self.fc2 = nn.Linear(conv_dim, 1) 


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class V(nn.Module):
    def __init__(self, conv_dim=64):
        super(V, self).__init__()
        self.fc1 = nn.Sequential( 
              nn.Linear(2048, conv_dim), 
              nn.LeakyReLU(negative_slope=0.2, inplace=True), 
              #nn.Dropout(p=0.2) 
             ) 
        self.fc2 = nn.Linear(conv_dim, 1) 


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x



s_feature = feature_network(source).detach()
t_feature = feature_network(target).detach()

#calculate the weight-opt

s_logit = full_connect_network(s_feature); t_logit = full_connect_network(t_feature)
S = attn_network(s_logit, t_logit)   

C = kernel_gram(s_feature,t_feature,s_feature.size()[0],z_t.size()[0],2048)       
CC = torch.zeros(C.size())
CC = Variable(CC).cuda()
CC.data.copy_(C.data)
epsilon = 1

CC = torch.mul(S,CC)

source_ot = ot_connect_network(s_feature)
target_ot = ot_connect_network(t_feature)
f = torch.exp((source_ot.repeat(1,target_ot.size()[0]) - CC)/epsilon)
g_c_e = - epsilon*torch.log(torch.mean(f))
W_loss = -torch.mean(g_c_e) + torch.mean(target_ot) - epsilon


  