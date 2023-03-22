_base_ = '../nonlocal_r50-d8_512x512_20k_voc12aug.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(depth=101),
    decode_head=dict(mode='embedded_gaussian', k=[0,8,0,8], qkv_bias=False))
