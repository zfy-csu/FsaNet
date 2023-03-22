'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
do 2D-DCT by P, Linsoftmax form, calculate KTQ first, 
high-frequency processed by 1X1 group conv
'''

import torch
import torch.nn as nn
import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
import timeit


def getP(H,W,k): #H=We
    ind = np.array([2*x+1 for x in range(H)])
    Dht = [np.sqrt(2)/np.sqrt(H)*np.cos(u*ind*np.pi/(2*H)) for u in range(H)]    
    Dht = torch.tensor(Dht, dtype=torch.float32)
    Dht[0,:] = 1/np.sqrt(H)#one row reoresent one frequency
    Dh = Dht.transpose(0,1).contiguous() ##one col reoresent one frequency
    Dh = Dh[:,k[0]:k[1]]
    
    ind = np.array([2*x+1 for x in range(W)])
    Dvt = [np.sqrt(2)/np.sqrt(H)*np.cos(u*ind*np.pi/(2*W)) for u in range(W)]    
    Dvt = torch.tensor(Dvt, dtype=torch.float32)
    Dvt[0,:] = 1/np.sqrt(W)#one row reoresent one frequency
    Dv = Dvt.transpose(0,1).contiguous() ##one col reoresent one frequency
    Dv = Dv[:,k[2]:k[3]]
    EH = torch.eye(H*W).reshape(H*W,H,W)
    P = EH.matmul(Dv).transpose(1,2).matmul(Dh).transpose(1,2).reshape(H*W, (k[1]-k[0])*(k[3]-k[2])) 
    return P.cuda()

  
class DCTNLAttention11(nn.Module):
    """ DCTNLAttention Module"""
    def __init__(self, in_dim, k, num_classes=None):
        super(DCTNLAttention11, self).__init__()
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
            self.P = getP(H, W, k)#HH,kk
        PT=self.P.transpose(0, 1)#kk,HH
        xP = x.view(B,C,H*W).matmul(self.P) #B,C,kk
        WqxP = self.query_conv(xP)#B,o,kk
        lamdq = torch.norm(WqxP.matmul(PT),dim=1,keepdim=True)#B,1,HH
        WkxP = self.key_conv(xP)#B,o,HW
        lamdk = torch.norm(WkxP.matmul(PT),dim=1,keepdim=True).view(B,H*W,1)#B,HH,1
        WvxP = self.value_conv(xP)#B,C,kk
        
        fatt = WkxP.transpose(1, 2).bmm(WqxP) #B,kk,kk
        lamdkP = self.P.div(lamdk)#B,HH,kk
        PTlamdq = PT.div(lamdq)#B,kk,HH
        lamdv = H*W + torch.sum(lamdkP,dim=1,keepdim=True).bmm(fatt).bmm(PTlamdq)#B,1,HH       
        
        
        out = PT.matmul(lamdkP).bmm(fatt).bmm(PTlamdq)+torch.sum(PT,dim=1,keepdim=True)#(B,kk,HH)+(kk,1)
        out = WvxP.bmm(out.div(lamdv))
        out = out + self.out_bias
        

        return self.gamma*(out.reshape(B,C,H,W)) + x
    
if __name__ == '__main__':
        B=1; in_dim = 512
        x = torch.randn(B,in_dim,97,97).cuda()
        model = DCTNLAttention11(in_dim,[0,5,0,5]).cuda()
        # %timeit out = model(x)
