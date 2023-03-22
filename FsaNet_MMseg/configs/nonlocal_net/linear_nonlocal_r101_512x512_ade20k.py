_base_ = [
    '../_base_/models/nonlocal_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(depth=101),
    decode_head=dict(num_classes=150, mode='linear_embedded_gaussian'), 
    auxiliary_head=dict(num_classes=150))
