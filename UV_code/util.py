'''
'''
import smtplib
import torch.nn.functional as F
import matplotlib  
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
        
def classification_accuracy(feature_network, full_connect_network, loader, use_gpu):
    correct = 0
    for data, label in loader:
        if use_gpu:
            data, label = data.cuda(), label.cuda()
        
        cache = feature_network(data)
        cache,_ = full_connect_network(cache)
        prob = F.softmax(cache, dim=1)
        pred = prob.data.max(dim=1)[1]
        if use_gpu:
            correct += pred.eq(label).cpu().sum().item()
        else:
            correct += pred.eq(label).sum().item()
    return correct / len(loader.dataset)

def plot_and_save(y, title, x_label, y_label, file_name):
    plt.plot(y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name+'.png')
    plt.clf()
    plt.close()

# network setting
class NetWork_Config:
    # universal config
    class_num = 31
    # resnet config
    resnet_name = 'ResNet50'
    fc_in_features = 2048
    bottleneck_dim = 512
    dropout_p = None
# training setting
class Train_Config:
    sigma = 1.0

    # semi learning config
    consistency_type = 'mse'
    threshold = 0.9

    #tradeoff
    lambda_1 = 1.0
    lambda_2 = 0.3
    lambda_3 = 0.5
    delta = 1


