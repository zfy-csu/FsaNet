_base_ = [
    '../../_base_/models/nonlocal_r50-d8.py',
    '../../_base_/datasets/pascal_voc12_aug.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21, mode='embedded_gaussian', k=[0,8,0,8], qkv_bias=False),
    auxiliary_head=dict(num_classes=21)
    )
