'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



#===============================================================================
# MMD loss
#===============================================================================

def calculate_Kij(iSample, jSample, sigma):
    return torch.exp(-torch.pow(torch.norm(iSample-jSample,p=2),exponent=2)/(2*sigma))


class MMD_loss(nn.Module):

    def __init__(self, name, train_config):
        super(MMD_loss, self).__init__()
        if name == 'unWeight':
            self.MMDLoss = _MMD_loss__A(train_config.sigma)
        elif name == 'Weight':
            self.MMDLoss = _MMD_loss__B(train_config.sigma)
        
    def forward(self, source_feature, source_label, target_feature, class_weight):
        return self.MMDLoss(source_feature, source_label, target_feature, class_weight)


class _MMD_loss__A(nn.Module):
    def __init__(self, sigma):
        super(_MMD_loss__A, self).__init__()
        self.sigma = sigma
        
    def forward(self, source_feature, source_label, target_feature, class_weight):
        ns = source_feature.size(0)
        nt = target_feature.size(0)
        
        first_part = 0
        for i in range(ns):
            for j in range(ns):
                first_part += calculate_Kij(source_feature[i],source_feature[j],self.sigma)
            
        sec_part = 0
        for i in range(nt):
            for j in range(nt):
                sec_part += calculate_Kij(target_feature[i],target_feature[j],self.sigma)
                
        third_part = 0
        for i in range(ns):
            for j in range(nt):
                third_part += calculate_Kij(source_feature[i],target_feature[j],self.sigma)
        
        return first_part/ns**2 + sec_part/nt**2 - 2*third_part/(ns*nt)


class _MMD_loss__B(nn.Module):

    def __init__(self, sigma):
        super(_MMD_loss__B, self).__init__()
        self.sigma = sigma
        
    def forward(self, source_feature, source_label ,target_feature, class_weight):
        ns = source_feature.size(0)
        nt = target_feature.size(0)
        n = 0
        for i in range(ns):
            try:
                n += class_weight[source_label[i]]
            except Exception as e:
                print(class_weight)
                print(source_label)
        
        first_part = 0
        for i in range(ns):
            for j in range(ns):
                first_part += class_weight[source_label[i]] * \
                              class_weight[source_label[j]] * \
                              calculate_Kij(source_feature[i],source_feature[j],self.sigma)
            
        sec_part = 0
        for i in range(nt):
            for j in range(nt):
                sec_part += calculate_Kij(target_feature[i],target_feature[j],self.sigma)
                
        third_part = 0
        for i in range(ns):
            for j in range(nt):
                third_part += class_weight[source_label[i]] * \
                              calculate_Kij(source_feature[i],target_feature[j],self.sigma)
        
        return first_part/n**2 + sec_part/nt**2 - 2*third_part/(n*nt)



#===============================================================================
# semi-learning loss
#===============================================================================


class semi_learning_unlabel_loss(nn.Module):
    
    def __init__(self):
        super(semi_learning_unlabel_loss, self).__init__()
        
    def forward(self, feature):
        prob = F.softmax(feature, dim=1)
        H = -(prob*torch.log(prob)).sum(dim=1)
        return H.mean()


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax)


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    #num_classes = input1.size()[1]
    #return torch.sum((input1 - input2)**2) / num_classes
    batch_size = input1.size()[0]
    return torch.sum((input1 - input2)**2) / batch_size

def EMD_loss(input1, input2):
    """Simple Implementation Version of Wasserstain distance
    Used like in the WGAN

    Note:
    - Return the distance between two domains' center and take the Positive part
    - Sends gradients to both input1 and input2.
    """
    dist = torch.mean(input1)-torch.mean(input2)
    return dist.abs()

def entropy_loss(input_f):
    """Simple Implementation Version of target domain entropy

    Note:
    - Return the entropy of one domain, expect to reduce the variance
    - Sends gradients to both input1 and input2.
    """
    ans = torch.mul(input_f, torch.log(input_f+0.001))
    mid = torch.sum(ans, dim=1)
    
    return -torch.mean(mid)

def coral_loss(input1, input2, gamma = 1e-3):
    ''' Implementation of coral covariances (which is regularized)
   
    Note:
    - Return the covariances of all features. That's [f*f]
    - Send gradients to both input1 and input2
    '''
	# First: subtract the mean from the data matrix
    batch_size = input1.shape[0].double()
    h_src = input1 - torch.mean(input1, dim=0)
    h_trg = input2 - torch.mean(input2, dim=0)
   
    cov_src = (1./(batch_size-1)) * torch.mm(torch.transpose(h_src), h_src)
    cov_trg = (1./(batch_size-1)) * torch.mm(torch.transpose(h_trg), h_trg)
    # Returns the Frobenius norm
    # The mean account for the factor 1/d^2
    return torch.mean(torch.pow(cov_src-cov_trg, 2))
   
def log_coral_loss(input1, input2, gamma=1e-3):
    ''' Implementation of eig version coral covariances (which is regularized)
   
    Note:
    - Return the covariances of all features. That's [f*f]
    - Send gradients to both input1 and input2
    '''
    #First: subtract the mean from the data matrix
    batch_size = input1.shape[0].double()
    h_src = input1 - torch.mean(input1, dim=0)
    h_trg = input2 - torch.mean(input2, dim=0)

    cov_src = (1./(batch_size-1)) * torch.mm(torch.transpose(h_src), h_src)
    cov_trg = (1./(batch_size-1)) * torch.mm(torch.transpose(h_trg), h_trg)
   
    #eigen decomposition
    e_src, v_src = torch.symeig(cov_src, eigenvectors=True)
    e_trg, v_trg = torch.symeig(cov_trg, eigenvectors=True)
   
	# Returns the Frobenius norm
    log_cov_src = torch.mm(e_src,  torch.mm(torch.diag(torch.log(e_src+0.0001)), torch.transpose(v_src)))
    log_cov_trg = torch.mm(e_trg,  torch.mm(torch.diag(torch.log(e_trg+0.0001)), torch.transpose(v_trg)))
   
    return torch.mean(torch.pow(log_cov_src-log_cov_trg, 2))
def class_entropy_loss(input_f):
    """Simple Implementation Version of target domain entropy

    Note:
    - Return the entropy of one domain, expect to reduce the variance
    - Sends gradients to both input1 and input2.
    """
    meanvec = torch.mean(input_f, dim=0, keepdim=True)
    mid = torch.mul(meanvec, torch.log(meanvec+0.00001))
    
    return -torch.mean(mid)

def l2dist(source, target):
    """Computes pairwise Euclidean distances in torch."""
    def flatten_batch(x):
      dim = torch.prod(torch.tensor(x.shape[1:]))
      return x.reshape([-1, dim])
    def scale_batch(x):
      dim = torch.prod(torch.tensor(x.shape[1:]))
      return x/dim.double().sqrt()
    def prepare_batch(x):
      return scale_batch(flatten_batch(x))

    target_flat = prepare_batch(target)  # shape: [bs, nt]
    target_sqnorms = torch.sum(target_flat.pow(2),1).unsqueeze(1)
    target_sqnorms_t = target_sqnorms.transpose(0,1)  # shape: [bs, nt]

    source_flat = prepare_batch(source)  # shape: [bs, ns]
    source_sqnorms = torch.sum(source_flat.pow(2),1).unsqueeze(1)

    dotprod = source_flat.mm(target_flat.transpose(0,1))
    sqdist = source_sqnorms - 2*dotprod + target_sqnorms_t #broadcast
    dist = F.relu(sqdist).sqrt()  # potential tiny negatives are suppressed
    return dist  # shape: [ns, nt]  

def Lifted_loss(feature_s, feature_t, y_s, y_t, alpha=80, beta=15, lamda=3, neg_pair_num=10):
    beta = torch.tensor(beta).cuda()

    #Define the between class distance
    D_b = l2dist(feature_s, feature_t) #b*b
    pair_matrix = y_s.mm(y_t.transpose(0,1))#b*b
    index = torch.eq(pair_matrix, 1) #b*b
    
    D_p = torch.masked_select(D_b, index) #b*b
    Pos = torch.max(D_p, torch.ones_like(D_p)*beta).reshape([-1,1]) #1
    #Define the within class distance
    s_index = torch.nonzero(index)[:,0]
    t_index = torch.nonzero(index)[:,1]
    pair_s = 1 - y_s.mm(y_s.transpose(0,1)).float()
    pair_t = 1 - y_t.mm(y_t.transpose(0,1)).float()
    
    D_ws = l2dist(feature_s, feature_s)
    D_wt = l2dist(feature_t, feature_t)
    s_A = torch.exp(alpha - D_ws).mul(pair_s)
    t_A = torch.exp(alpha - D_wt).mul(pair_t)
    
    neg_pair_num = min(neg_pair_num, s_A.size()[0])
   # D_ws = torch.sum(torch.topk(s_A, neg_pair_num, 1)[0], 1)
    D_ws = torch.sum(s_A,1)
    D_wt = torch.sum(t_A,1)
   # D_wt = torch.sum(torch.topk(t_A, neg_pair_num, 1)[0], 1)
    Neg = torch.log(D_ws[s_index] + D_wt[t_index] + 0.0001).reshape([-1,1])
    
    J =  lamda * Neg + Pos
    loss = torch.mean(torch.max(torch.zeros_like(J), J))
    return loss

