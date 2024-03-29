_base_ = [
    '../../_base_/models/nonlocal_r50-d8.py', '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=150, mode='embedded_gaussian', k=[0,8,0,8], qkv_bias=False), 
    auxiliary_head=dict(num_classes=150))
