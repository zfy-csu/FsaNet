## FsaNet based on mmsegmentation
# To add our FsaNet to mmsegmentation, please:
1. Put 'fsa_head1d.py' into 'mmsegmentation-master/mmseg/models/decode_heads'
2. Add 'from .fsa_head1d import FSAHead1d' to mmseg/models/decode_heads/__init__.py
3. Merge our 'configs/fsanet' folder to origin configs folder
4. Follow the instructions of mmsegmentation and run our FsaNet by config files in 'configs/fsanet'. For example, you can run:
```bash
$ export CUDA_VISIBLE_DEVICES=0,1,2,3
$ python -m torch.distributed.launch --nproc_per_node=4 --master_port=25904 ./tools/train.py configs/fsanet/r2_fsa1d_dot_r101-d8_769x769_80k_cityscapes.py --launcher pytorch
$ python tools/test.py configs/fsanet/r1_fsa1d_lin_r101-d8_769x769_80k_cityscapes.py checkpoints/r2_fsa1d_lin_r101-d8_769x769_80k_cityscapes.pth --eval cityscapes
```
#To verify the Non_local Embedded Gaussian Module is approximate to linear attention adpatively, please:
1. Change 'mmsegmentation-master/mmseg/models/decode_heads/nl_head.py' to our nl.head.py
2. Change '/home/XXX/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/cnn/bricks/non_local.py' to our 'non_local.py'
3. Merge the 'configs/nonlocal_net' folder to origin configs folder: 
 Then you can obeserve that this cause small diffenrence by running:
```bash
$ python tools/test.py configs/nonlocal_net/nonlocal_r101_769x769_cityscapes.py checkpoints/nonlocal_r101-d8_769x769_80k_cityscapes_20200607_183428-0e1fa4f9.pth --eval cityscapes #79.3
$ python tools/test.py configs/nonlocal_net/linear_nonlocal_r101_769x769_cityscapes.py checkpoints/nonlocal_r101-d8_769x769_80k_cityscapes_20200607_183428-0e1fa4f9.pth --eval cityscapes #79.3
```
```bash
$ python tools/test.py configs/nonlocal_net/nonlocal_r101_512x512_ade20k.py checkpoints/nonlocal_r101-d8_512x512_160k_ade20k_20210827_221502-7881aa1a.pth --eval mIoU #44.62
$ python tools/test.py configs/nonlocal_net/linear_nonlocal_r101_512x512_ade20k.py checkpoints/nonlocal_r101-d8_512x512_160k_ade20k_20210827_221502-7881aa1a.pth --eval mIoU #44.43
```
```bash
$ python tools/test.py configs/nonlocal_net/nonlocal_r101_512x512_voc12aug.py checkpoints/nonlocal_r101-d8_512x512_40k_voc12aug_20200614_000028-7e5ff470.pth --eval mIoU #78.24
$ python tools/test.py configs/nonlocal_net/linear_nonlocal_r101_512x512_voc12aug.py checkpoints/nonlocal_r101-d8_512x512_40k_voc12aug_20200614_000028-7e5ff470.pth --eval mIoU #77.2
```
#To verify frequency ablation study on non_local net, please merge the 'configs/nonlocal_net' folder to origin configs folder.
#8.py means k is [0,8,0,8], you can change k by reset k. For example, you can run:
1. For cityscapes:
```bash
$ python tools/test.py configs/nonlocal_net/nonlocal_r50-d8_769x769_40k_cityscapes/8.py checkpoints/nonlocal_r50-d8_769x769_40k_cityscapes_20200530_045243-82ef6749.pth --eval cityscapes #78.0
$ python tools/test.py configs/nonlocal_net/nonlocal_r50-d8_769x769_80k_cityscapes/8.py checkpoints/nonlocal_r50-d8_769x769_80k_cityscapes_20200607_193506-1f9792f6.pth --eval cityscapes #78.8
$ python tools/test.py configs/nonlocal_net/nonlocal_r101-d8_769x769_40k_cityscapes/8.py checkpoints/nonlocal_r101-d8_769x769_40k_cityscapes_20200530_045348-8fe9a9dc.pth --eval cityscapes #78.5
$ python tools/test.py configs/nonlocal_net/nonlocal_r101-d8_769x769_80k_cityscapes/8.py checkpoints/nonlocal_r101-d8_769x769_80k_cityscapes_20200607_183428-0e1fa4f9.pth --eval cityscapes #79.3
```
2. For ADE20K:
```bash
$ python tools/test.py configs/nonlocal_net/nonlocal_r50-d8_512x512_80k_ade20k/8.py checkpoints/nonlocal_r50-d8_512x512_80k_ade20k_20200615_015801-5ae0aa33.pth --eval mIoU  #40.34
$ python tools/test.py configs/nonlocal_net/nonlocal_r50-d8_512x512_160k_ade20k/8.py checkpoints/nonlocal_r50-d8_512x512_160k_ade20k_20200616_005410-baef45e3.pth --eval mIoU #41.51
$ python tools/test.py configs/nonlocal_net/nonlocal_r101-d8_512x512_80k_ade20k/8.py checkpoints/nonlocal_r101-d8_512x512_80k_ade20k_20200615_015758-24105919.pth --eval mIoU  #42.59
$ python tools/test.py configs/nonlocal_net/nonlocal_r101-d8_512x512_160k_ade20k/8.py checkpoints/nonlocal_r101-d8_512x512_160k_ade20k_20210827_221502-7881aa1a.pth --eval mIoU #44.06
```
3. For VOC12aug:
```bash
$ python tools/test.py configs/nonlocal_net/nonlocal_r50-d8_512x512_20k_voc12aug/8.py checkpoints/nonlocal_r50-d8_512x512_20k_voc12aug_20200617_222613-07f2a57c.pth --eval mIoU #75.53
$ python tools/test.py configs/nonlocal_net/nonlocal_r50-d8_512x512_40k_voc12aug/8.py checkpoints/nonlocal_r50-d8_512x512_40k_voc12aug_20200614_000028-0139d4a9.pth --eval mIoU #76.52
$ python tools/test.py configs/nonlocal_net/nonlocal_r101-d8_512x512_20k_voc12aug/8.py checkpoints/nonlocal_r101-d8_512x512_20k_voc12aug_20200617_222615-948c68ab.pth --eval mIoU #77.71
$ python tools/test.py configs/nonlocal_net/nonlocal_r101-d8_512x512_40k_voc12aug/8.py checkpoints/nonlocal_r101-d8_512x512_40k_voc12aug_20200614_000028-7e5ff470.pth --eval mIoU #77.9
```
## Thanks to the Third Party Libs
https://github.com/open-mmlab/mmsegmentation
