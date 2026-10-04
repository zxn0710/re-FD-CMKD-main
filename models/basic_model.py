import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet34, resnet101

class AClassifier(nn.Module):
    def __init__(self, args):
        super(AClassifier, self).__init__()
        if args.dataset == 'VGGSound':
            n_classes = 50
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.net = resnet18(modality='audio')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, audio):
        pre_a,a = self.net(audio)
        pre_a = F.adaptive_avg_pool2d(pre_a, 1)
        pre_a = torch.flatten(pre_a, 1)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        out = self.classifier(a)
        return pre_a,out


class VClassifier(nn.Module):
    def __init__(self, args):
        super(VClassifier, self).__init__()
        if args.dataset == 'VGGSound':
            n_classes = 50
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.net = resnet18(modality='visual')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, visual, B):
        pre_v,v = self.net(visual)
        (_, C, H, W) = v.size()
        pre_v = pre_v.view(B, -1, C, H, W)
        pre_v = pre_v.permute(0, 2, 1, 3, 4)
        pre_v = F.adaptive_avg_pool3d(pre_v, 1)
        pre_v = torch.flatten(pre_v, 1)
        
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        out = self.classifier(v)
        return pre_v,out
    