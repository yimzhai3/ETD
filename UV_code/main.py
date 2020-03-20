#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:23:24 2019

@author: zym
"""
import os
import pandas as pd
from util import NetWork_Config, Train_Config
import torch
import numpy as np
import torch.cuda
torch.cuda.set_device(6)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange
from scipy.io import savemat
from data import DataLoad
from network import Feature, Full_Connect, weights_init, U, V, Attention
from loss import entropy_loss
from Wasserstain_dist import distance
from util import classification_accuracy

#+++configure.csv+++++++++++++++++++++++++++++++++
# load configure file
df = pd.read_csv('UV.csv')
df = df.dropna(axis=0,how='all')
#+++++++++++++++++++++++++++++++++++++++++++++++++
# initial setting variable
network_config = NetWork_Config()
train_config = Train_Config()
# basic setting
use_gpu = False
if torch.cuda.is_available():
    use_gpu = True
def reset_grad():
    fea.zero_grad()
    pre.zero_grad()
    u.zero_grad()
    v.zero_grad()
    attn.zero_grad()  
#===============================================================================
# loop configure training
#===============================================================================
# loop configure
for iConfig in df.index.tolist():
    experiment_name = df['experiment_name'][iConfig]
    source_domain = df['source_domain'][iConfig]
    target_domain = df['target_domain'][iConfig]
    network_config.class_num = int(df['class_num'][iConfig])
    network_config.resnet_name = df['resnet_name'][iConfig]
    network_name = df['network_name'][iConfig] 
    epochs = int(df['epochs'][iConfig])
    Pretrain_Epoch = int(df['Pretrain_Epoch'][iConfig])
    train_batch_size = int(df['train_batch_size'][iConfig])
    
    lr = float(df['lr'][iConfig])
    lr_feature = float(df['lr_feature'][iConfig])
    lr_fc = float(df['lr_fc'][iConfig])
    beta1 = float(df['beta1'][iConfig])
    beta2 = float(df['beta2'][iConfig])  
    epsilon = float(df['epsilon'][iConfig])
    lambda_1 = float(df['lambda_1'][iConfig])
    lambda_2 = float(df['lambda_2'][iConfig])

    network_config.fc_in_features = int(df['fc_in_features'][iConfig])
    network_config.bottleneck_dim = int(df['bottleneck_dim'][iConfig])
    network_config.dropout_p = None if df['dropout_p'][iConfig]==0 else float(df['dropout_p'][iConfig])        
    #+++result save path+++++++++++++++++++++++++++++++++++
    file_path = '.'+os.path.sep+'UV' 
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    experiment_base_path = '.'+os.path.sep+'UV'+os.path.sep+experiment_name        
    if not os.path.exists(experiment_base_path):
        os.mkdir(experiment_base_path)
    log_file = open(file=os.path.join(experiment_base_path,'{}_{}_log.txt'.format(source_domain,target_domain)), mode='w')
    best_result_file = open(file=os.path.join(experiment_base_path,'{}_{}_best_result.txt'.format(source_domain,target_domain)), mode='w')  
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    
    # ===============================================================================
    # Load data
    # ===============================================================================
    DataLoader = DataLoad(batch_size=train_batch_size, num_workers=4)
    source_loader = DataLoader.source_loader(source_domain)
    target_loader = DataLoader.target_loader(target_domain)       
    # ===============================================================================
    # initial network
    # ===============================================================================
    fea = Feature(network_name, network_config)
    pre = Full_Connect(network_name, network_config)
    pre.apply(weights_init)
    u = U(512)
    v = V(512)
    u.apply(weights_init)
    v.apply(weights_init)
    attn = Attention(in_dim=network_config.class_num, out_dim=network_config.class_num)
    attn.apply(weights_init)
    
    if use_gpu:
        fea.cuda()
        pre.cuda()   
        u.cuda()
        v.cuda()
        attn.cuda()     
    # ===============================================================================
    # loss function
    # ===============================================================================
    class_weight = torch.Tensor(np.array([1.0] * network_config.class_num))
    class_criterion = nn.CrossEntropyLoss(class_weight)
    entropy_criterion = entropy_loss       
    if use_gpu:
        class_weight = class_weight.cuda()
        class_criterion = class_criterion.cuda()   
    # ===============================================================================
    # optimizer
    # ===============================================================================
    optimizer_initial = optim.Adam([{'params': fea.parameters(), 'lr': lr_feature},
                                    {'params': pre.parameters(), 'lr': lr_fc},
                                     {'params': attn.parameters(), 'lr': lr_feature}],
                                   lr, [beta1, beta2], weight_decay=0.01)
    
    optimizer_ot = optim.Adam([{'params': u.parameters()},
                                {'params': v.parameters()}],
                                lr, [beta1, beta2], weight_decay=0.01)
    # ===============================================================================
    # pre-train
    # ===============================================================================
    # initial variable
    classifier_loss_list = []
    epoch_loss_list = []
    source_acc_list = []
    target_acc_list = []
    best_pretrain_acc = 0
    optimizer = optimizer_initial
    for _ in trange(Pretrain_Epoch, desc="Pre-train Epoch", unit='epoch'):
        fea.train(mode=True)
        pre.train(mode=True)
        loss_temp = []
        for source, source_label in source_loader:
            source, source_label = Variable(source), Variable(source_label)
            if use_gpu:
                source, source_label = source.cuda(), source_label.cuda()
            fea.zero_grad()
            pre.zero_grad()
            # forward
            source_y, _ = pre(fea(source))
            # calculate loss
            classifier_loss = class_criterion(source_y, source_label)
            classifier_loss.backward()
            classifier_loss_list.append(classifier_loss.cpu().detach().numpy().tolist())
            loss_temp.append(classifier_loss.cpu().detach().numpy())
            optimizer.step()
        epoch_loss_list.append(np.mean(loss_temp))
        fea.eval()
        pre.eval()
        # accuracy on source
        source_acc = classification_accuracy(fea, pre, source_loader, use_gpu)
        source_acc_list.append(source_acc)
        # accuracy on target
        target_acc = classification_accuracy(fea, pre, target_loader, use_gpu)
        target_acc_list.append(target_acc)
        if target_acc > best_pretrain_acc:
            best_pretrain_acc = target_acc
        info ='[Pre-train] Source accuracy: {:.4f}  Target accuracy: {:.4f} Best Pretrain accuracy: {:.4f} Loss:{:.4f}'.format(source_acc, target_acc, best_pretrain_acc,np.mean(loss_temp))
        print(info)
        log_file.write(info+'\n')
    pretrain_last_acc = target_acc
    savemat(experiment_base_path+os.path.sep+'{}_{}_pre_train'.format(source_domain,target_domain),
            {'loss': classifier_loss_list,'epoch_loss':epoch_loss_list,'source_acc': source_acc_list, 'target_acc': target_acc_list})
    # ===============================================================================
    # ot-train
    # ===============================================================================
    # initial variable
    classifier_loss_list = []
    ot_loss_list = []
    source_acc_list = []
    target_acc_list = []
    best_ot_acc = 0              
    optimizer = optimizer_initial
    optimizer_ot = optimizer_ot
    for _ in trange(15, desc="ot-train Epoch", unit='epoch'):
        fea.train(mode=True)
        pre.train(mode=True)
        u.train(mode=True)
        v.train(mode=True)
        attn.train(mode=True)
        loss_ot_temp = []
        for (source, source_label), (target, _) in zip(source_loader, target_loader):
            source, source_label = Variable(source), Variable(source_label)
            if use_gpu:
                source, source_label = source.cuda(), source_label.cuda()
                target = target.cuda()
            # set gradient to zero
            reset_grad()
            
            s_feature = fea(source) ; t_feature = fea(target)  
            C = distance(s_feature,t_feature)
            C = Variable(C).cuda()        
            
            s_logit, _= pre(s_feature) ; t_logit,_ = pre(t_feature)
            weight = attn(s_logit, t_logit)
            C =  weight.mul(C)
            
            for i in range(5):                        
                d_s = u(s_feature.detach())
                d_t = v(t_feature.detach())
                f = torch.exp((d_s.repeat(1,d_t.size()[0]) + d_t.repeat(1,d_s.size()[0]).t() - C)/epsilon)
                ot_Loss = -(torch.mean(d_s) + torch.mean(d_t) - epsilon*torch.mean(f))
                ot_Loss.cuda()
                ot_Loss.backward(retain_graph=True)
                optimizer_ot.step()
                reset_grad()
            #-------------------------------------------------------------------------------
            source_y, _ = pre(s_feature)
            cache, _ = pre(t_feature)
            proba = F.softmax(cache, dim=1)
            entroLoss = entropy_criterion(proba)
            classifier_loss = class_criterion(source_y, source_label)
            Total_loss = classifier_loss + lambda_2*entroLoss 
            Total_loss.backward()
            loss_ot_temp.append(Total_loss.cpu().detach().numpy())
            # optimize
            optimizer_ot.step()
        ot_loss_list.append(np.mean(loss_ot_temp))    
        # set evaluate mode
        fea.eval()
        pre.eval()
        # accuracy on source
        source_acc = classification_accuracy(fea, pre, source_loader, use_gpu)
        source_acc_list.append(source_acc)
        # accuracy on target
        target_acc = classification_accuracy(fea, pre, target_loader, use_gpu)
        target_acc_list.append(target_acc)
        if target_acc > best_ot_acc:
            best_ot_acc = target_acc
        info ='[ot-train] Source accuracy: {:.4f}  Target accuracy: {:.4f} Best ot accuracy: {:.4f} ot_loss: {:.4f}'.format(source_acc, target_acc, best_ot_acc,np.mean(loss_ot_temp))
        print(info)
        log_file.write(info+'\n')
        
    # ===============================================================================
    # train network
    # ===============================================================================  
    source_acc_list = []
    target_acc_list = []
    Total_loss_list = []
    epoch_loss_list = []
    optimizer = optimizer_initial
    optimizer_ot = optimizer_ot
    
    best_origin_acc = 0     
    for i in trange(epochs, desc="Total Epoch", unit='epoch'):
        fea.train(mode=True)
        pre.train(mode=True)
        u.train(mode=True)
        v.train(mode=True)
        attn.train(mode=True)
        loss_temp = []
        for (source, source_label), (target, _) in zip(source_loader, target_loader):
            source, source_label = Variable(source), Variable(source_label)
            if use_gpu:
                source, source_label = source.cuda(), source_label.cuda()
                target = target.cuda()
            reset_grad()  
            s_feature = fea(source) ; t_feature = fea(target)            
            C = distance(s_feature,t_feature)
            C  = Variable(C).cuda()
            s_logit,_ = pre(s_feature) ; t_logit,_ = pre(t_feature)
            weight = attn(s_logit, t_logit)
            C =  weight.mul(C)
            for k in range(10):                     
                d_s = u(s_feature.detach())
                d_t = v(t_feature.detach())
                f = torch.exp((d_s.repeat(1,d_t.size()[0]) + d_t.repeat(1,d_s.size()[0]).t() - C)/epsilon)
                ot_Loss = -(torch.mean(d_s) + torch.mean(d_t) - epsilon*torch.mean(f))          
                ot_Loss.backward(retain_graph=True)
                optimizer_ot.step()
                reset_grad() 
            source_y,_ = pre(s_feature)
            cache,_ = pre(t_feature)
            proba = F.softmax(cache, dim=1)
            entroLoss = entropy_criterion(proba)
            
            d_s = u(s_feature)
            d_t = v(t_feature)
            
            f = torch.exp((d_s.repeat(1,d_t.size()[0]) + d_t.repeat(1,d_s.size()[0]).t() - C)/epsilon)
            WLoss = (torch.mean(d_s) + torch.mean(d_t) - epsilon*torch.mean(f))
            if use_gpu:
                WLoss.cuda()
            classifier_loss = class_criterion(source_y, source_label)
            
            Total_loss = classifier_loss + lambda_1*(WLoss) + lambda_2*entroLoss #+ lambda_3*entroLoss2
            Total_loss.backward()
            optimizer.step()
            
            Total_loss_list.append(Total_loss.cpu().detach().numpy().tolist())
            loss_temp.append(Total_loss.cpu().detach().numpy())
            
        epoch_loss_list.append(np.mean(loss_temp))
        fea.eval()
        pre.eval()
        # accuracy on source
        source_acc = classification_accuracy(fea, pre, source_loader, use_gpu)
        source_acc_list.append(source_acc)
        # accuracy on target
        target_acc = classification_accuracy(fea, pre, target_loader, use_gpu)
        target_acc_list.append(target_acc)

        if target_acc > best_origin_acc:
            best_origin_acc = target_acc
            best_step = i
            torch.save({'Fea_net': fea.state_dict(),
                        'Classifier': pre.state_dict(),
                        'best_acc': best_origin_acc,
                        'best_step': best_step,
                        }, os.path.join(experiment_base_path, "{}_{}_best_origin_network.pth.tar".format(source_domain,target_domain)))
        info ="Source_acc:{:.4f}  Target_acc:{:.4f} loss:{:.4f}".format(source_acc, target_acc,np.mean(loss_temp)) 
        print(info)
        log_file.write(info+'\n')
    info = "Pretrain_last:{:.4f} Best_pretrain:{:.4f} Best_origin:{:.4f} ".format(pretrain_last_acc, best_pretrain_acc, best_origin_acc)
    print(info)
    best_result_file.write(info+'\n')  
    savemat(experiment_base_path+os.path.sep+'{}_{}_main_train'.format(source_domain,target_domain),
            {'loss': Total_loss_list, 'epoch_loss': epoch_loss_list,'source_acc': source_acc_list, 'target_acc': target_acc_list})    
    log_file.close()
    best_result_file.close()   
    #===========================================================================
    # release GPU memory
    #===========================================================================   
    del DataLoader
    del source_loader
    del target_loader
    del fea
    del pre
    del attn
    del weight
    del u
    del v    
    del class_weight
    del class_criterion
    del entropy_criterion   
    
    del optimizer_initial
    del optimizer

    del source
    del source_label
    del target
    del cache
    del source_y
    del classifier_loss
    del s_feature
    del t_feature
    del ot_Loss
    del WLoss
   
    del Total_loss  
    torch.cuda.empty_cache()
        
    
