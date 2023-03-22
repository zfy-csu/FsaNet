import torch
import torch.nn as nn
from torch.nn import Softmax
#import torchvision.transforms as transforms
#from PIL import Image
#import timeit


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, num_classes=None):
        super(CrissCrossAttention,self).__init__()
        out_dim = max(in_dim//8, 2) 
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
      
    def forward(self, x, fix_value0=None):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)#B,o,H,W
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)#BW,H,out_dim
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)#BH,W,out_dim
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)#BW,out_dim,H
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)#BH,out_dim,W
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)#BW,C,H
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)#BH,C,W
        
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)#B,H,W,H
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width) #B,H,W,W
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))#B,H,W,H+W #https://github.com/pytorch/pytorch/issues/6864

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        out = out_H + out_W
        
        return self.gamma*(out) + x

    
if __name__ == '__main__':
    B=1; in_dim = 512
    x = torch.randn(B,in_dim,97,97).cuda()
    model = CrissCrossAttention(in_dim).cuda() 
    # %timeit out = model(x)
