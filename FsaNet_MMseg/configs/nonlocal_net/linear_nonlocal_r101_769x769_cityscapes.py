_base_ = [
    '../_base_/models/nonlocal_r50-d8.py',
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(depth=101),
    decode_head=dict(mode='linear_embedded_gaussian', align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
