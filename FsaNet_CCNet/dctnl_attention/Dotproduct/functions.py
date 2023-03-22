'''
do 2D-DCT by P, scaled Dotproduct form, 
calculated in image space, Simplified calculation is realized only by the free matrix associative law
'''

import torch
import torch.nn as nn
import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import timeit
import torch

def getP(H,k): #H=W
    ind = np.array([2*x+1 for x in range(H)])
    Dt = [np.sqrt(2)/np.sqrt(H)*np.cos(u*ind*np.pi/(2*H)) for u in range(H)]    
    Dt = torch.tensor(Dt, dtype=torch.float32)
    Dt[0,:] = 1/np.sqrt(H)#one row reoresent one frequency
    D = Dt.transpose(0,1).contiguous() ##one col reoresent one frequency
    Dh = D[:,k[0]:k[1]]
    Dv = D[:,k[2]:k[3]]
    EH = torch.eye(H**2).reshape(H*H,H,H)
    P = EH.matmul(Dh).transpose(1,2).matmul(Dv).transpose(1,2).reshape(H**2, (k[1]-k[0])*(k[3]-k[2])) 
    return P.cuda()

  
class DCTNLAttention21(nn.Module):
    """ DCTNLAttention Module"""
    def __init__(self, in_dim, k, num_classes=None):
        super(DCTNLAttention21, self).__init__()
        out_dim = max(in_dim//8, 2)#zfy
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1,bias=False)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1,bias=False)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,bias=False)
        self.out_bias = nn.Parameter(torch.zeros(1,in_dim,1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.P = None
        self.k = k

    def forward(self, x):   
        B, C, H, W = x.size()
        k = self.k
        if self.P == None:
            self.P = getP(H, k)#HH,kk
        PT=self.P.transpose(0, 1)#kk,HH
        
        xP = x.view(B,C,H*W).matmul(self.P) #B,C,kk
        WqxP = self.query_conv(xP)#B,o,kk
        WkxP = self.key_conv(xP)#B,o,kk
        WvxP = self.value_conv(xP)#B,C,kk        
        fatt = WkxP.transpose(1, 2).bmm(WqxP)/(H*W) #B,kk,kk  
        
        out = WvxP.bmm(fatt).matmul(PT)
        out = out + self.out_bias

        return self.gamma*(out.reshape(B,C,H,W)) + x
    
if __name__ == '__main__':
  
    B=1; in_dim = 512
    x = torch.randn(B,in_dim,97,97).cuda()
    model = DCTNLAttention21(in_dim,[0,5,0,5]).cuda()
    # %timeit out = model(x)
