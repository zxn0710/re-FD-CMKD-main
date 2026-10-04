import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def log(tensor):
    result_tensor = torch.zeros_like(tensor)
    positive_mask = tensor > 0
    negative_mask = tensor < 0
    result_tensor[positive_mask] = torch.log1p(tensor[positive_mask])
    result_tensor[negative_mask] = -torch.log1p(torch.abs(tensor[negative_mask]))
    return result_tensor
    
def adjust_lr(lr=1e-2, iter=None, max_iter=100, power=0.9, optimizer=None):
    cur_lr = 1e-4 + (lr - 1e-4)*((1-float(iter)/max_iter)**(power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

def norm_feat(x):
    x_norms = torch.norm(x, p=2, dim=1, keepdim=True)
    x_normalized_features = x / x_norms.clamp(min=1e-8)
    return x_normalized_features

class FDFilter(nn.Module):
    def __init__(self):
        super(FDFilter, self).__init__()
        self.filter1 = nn.Parameter(torch.zeros(257), requires_grad=False)
        self.filter2 = nn.Parameter(torch.zeros(257), requires_grad=False)
        self.init_filter()

    def init_filter(self):
        self.filter1.data[1:129] = 1
        self.filter2.data[129:] = 1
        
    def forward(self, x):
        freq_domain = torch.fft.rfft(x, dim=1)

        binary_filter1 = (self.filter1 >= 0.5).float()
        filtered_freq_domain1 = freq_domain * binary_filter1
        binary_filter2 = (self.filter2 >= 0.5).float()
        filtered_freq_domain2 = freq_domain * binary_filter2

        low_freq_features1 = torch.fft.irfft(filtered_freq_domain1, dim=1).real
        low_freq_features2 = torch.fft.irfft(filtered_freq_domain2, dim=1).real
        return low_freq_features1,low_freq_features2

class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        x = F.relu(x)
        x = self.fc(x)
        return x