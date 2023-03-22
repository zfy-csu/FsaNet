#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:07:48 2022

@author: fengyuzhang
"""

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn.utils import constant_init, normal_init
from mmcv.cnn.bricks.conv_module import ConvModule
from ..builder import HEADS
from .fcn_head import FCNHead


def DCTBases(N,k):
    ind = np.array([2*x+1 for x in range(N)])
    D = [np.sqrt(2)/np.sqrt(N)*np.cos(u*ind*np.pi/(2*N)) for u in range(N)]    
    D = torch.tensor(D, dtype=torch.float32)#每行为相同频率,Row*Dt or D*Col  
    D[0,:] = 1/np.sqrt(N)
    D = D[k[0]:k[1],:]
    D = D.transpose(0, 1)#每列为相同频率,Row*D or Dt*Col
    return D.cuda()
def getP(H,W,k): #H=We
    if k[1] == -1:
        k[1] = H
    if k[3] == -1:
        k[3] = W    
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
    P = EH.matmul(Dv).transpose(1,2).matmul(Dh).transpose(1,2).reshape(H*W, -1) #(k[1]-k[0])*(k[3]-k[2])
    return P.cuda()
   
class FreNonLocal(nn.Module):
    """Frequency NonLocal Blocks.

    Args:
        temperature (float): Temperature to adjust attention. Default: 0.05
    """

    def __init__(self, 
                 in_channels,
                 reduction=2,
                 use_scale=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='dot_product',
                 k=None,
                 qkv_bias=True,
                 **kwargs):
        super().__init__()
        self.P = None
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode
        self.k = k 
        self.P = None

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            bias=qkv_bias)
        print('gbias',self.g.with_bias)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            bias=qkv_bias)
        self.phi = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            bias=qkv_bias)
        print('theta-bias',self.theta.with_bias)
        print('phi-bias',self.phi.with_bias)

        self.init_weights(**kwargs)
    def init_weights(self, std=0.01, zeros_init=True):
        if self.mode != 'gaussian':
            for m in [self.g, self.theta, self.phi]:
                normal_init(m.conv, std=std)
        else:
            normal_init(self.g.conv, std=std)
        if zeros_init:
            if self.conv_out.norm_cfg is None:
                constant_init(self.conv_out.conv, 0)
            else:
                constant_init(self.conv_out.norm, 0)
        else:
            if self.conv_out.norm_cfg is None:
                normal_init(self.conv_out.conv, std=std)
            else:
                normal_init(self.conv_out.norm, std=std)        

    def forward(self, x):
        B,C,H,W=x.size()     
        if self.P is None:           
            P = getP(H, W, self.k)
        else:
            P = self.P
        PT=P.transpose(0, 1)
        xP=x.view(B,C,-1).matmul(P)
        
        if self.mode == 'dot_product':
            WqxP = self.theta(xP)/self.inter_channels**0.5 if self.use_scale else self.theta(xP) #B,o,kk
            WkxP = self.phi(xP)#B,o,kk
            WvxP = self.g(xP)#B,C,kk        
            fatt = WkxP.transpose(1, 2).bmm(WqxP)/(H*W) #B,kk,kk  
            out = WvxP.bmm(fatt).matmul(PT)
        else: #self.mode == 'linsoftmax':
            eps = 1e-12
            WqxP = self.theta(xP)/self.inter_channels**0.5 if self.use_scale else self.theta(xP) #B,o,kk
            lamdq = torch.norm(WqxP.matmul(PT),dim=1,keepdim=True)+eps#B,1,HW
            WkxP = self.phi(xP)#B,o,kk
            lamdk = torch.norm(WkxP.matmul(PT),dim=1,keepdim=True).view(B,H*W,1)+eps#B,HW,1
            WvxP = self.g(xP)#B,o,kk 
            fatt = WkxP.transpose(1, 2).bmm(WqxP) #B,kk,kk
            lamdkP = P.div(lamdk)#B,HW,kk
            PTlamdq = PT.div(lamdq)#B,kk,HW
            lamdv = H*W + torch.sum(lamdkP,dim=1,keepdim=True).bmm(fatt).bmm(PTlamdq)+eps#B,1,HW       
            out = PT.matmul(lamdkP).bmm(fatt).bmm(PTlamdq)+torch.sum(PT,dim=1,keepdim=True)#(B,kk,HW)+(kk,1)
            out = WvxP.bmm(out.div(lamdv))

        out = self.conv_out(out).reshape(B,C,H,W)
        output = x + out
        
        return output
        
@HEADS.register_module()
class FSAHead1d(FCNHead):
    """Non-local Neural Networks.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): no use.
        mode (str): The nonlocal mode. Options are 'dot_product',
            'linsoftmax'. Default: 'dot_product.'.
    """

    def __init__(self,
                 reduction=2,
                 use_scale=False,
                 qkv_bias=False,
                 recurrence=1,
                 k=[0,5,0,5],
                 mode='dot_product',
                 **kwargs):
        super(FSAHead1d, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale=use_scale
        self.mode = mode    
        self.qkv_bias=qkv_bias
        self.recurrence = recurrence
        self.conv_cfg = dict(type='Conv1d')
        print('fsa1d recurrence is', self.recurrence)
        print('fsa1d k is', k)
        self.nl_block = FreNonLocal(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            qkv_bias=self.qkv_bias,
            k=k
            )

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        for i in range(self.recurrence):
            output = self.nl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
