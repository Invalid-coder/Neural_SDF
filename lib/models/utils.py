import torch
import torch.nn as nn
import numpy as np

def init_decoder(net, args):
    for k, v in net.named_parameters():
        if 'weight' in k:
            if args is not None and args.periodic:
                std = np.sqrt(6 / v.shape[0])
                if k == 'l1.weight':
                    std = std * 30.0
            else:
                std = np.sqrt(2) / np.sqrt(v.shape[0])
            nn.init.normal_(v, 0.0, std)
        if 'bias' in k:
            nn.init.constant_(v, 0)
        if k == 'lout.weight' or ('lout' in k and 'weight' in k):
            if args is not None and args.periodic:
                std = np.sqrt(6 / v.shape[1])
                nn.init.constant_(v, std)
            else:
                std = np.sqrt(np.pi) / np.sqrt(v.shape[1])
                nn.init.constant_(v, std)
        if k == 'lout.bias' or ('lout' in k and 'bias' in k):
            if args is not None and args.periodic:
                nn.init.constant_(v, 0)
            else:
                nn.init.constant_(v, -1)

    return net
