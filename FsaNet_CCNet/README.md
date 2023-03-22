## FsaNet based on codes of CCNet
## Requirements
*****************************************************************
To check the SOTA result compared to other ResNet101-based self-attention networks, you just need Python 3, Pytorch 1.4, OpenCV and PIL.<br>
```bash
# Install **Pytorch-1.4**
$ conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
```
## Training and Evaluation (DCTNLAttention21 corresponds to Dotproduct, DCTNLAttention11 corresponds to LinSoftmax)
ImageNet Pre-trained Model can be downloaded from [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth).
```bash
$ export CUDA_VISIBLE_DEVICES=0,1,2,3
$ python train.py --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 0,1,2,3 --learning-rate 0.01 --input-size 849,849 --weight-decay 0.0001 --batch-size 8 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model fsanet --att-mode DCTNLAttention21 --recurrence 2 --k [0,8,0,8] --data-name CSDataSet  --data-list dataset/list/cityscapes/train.lst --data-dir dataset/cityscapes

$ python evaluate.py --gpu=0 --data-dir dataset/cityscapes/ --model fsanet --att-mode DCTNLAttention21 --recurrence 2 --k [0,8,0,8] --restore-from './snapshots/r2dot8305.pth' --batch-size 1 --experimentID r2CityDot

$python test.py --model fsanet --att-mode DCTNLAttention21 --recurrence 2 --k [0,8,0,8] --restore-from './snapshots/r2dot8305.pth' --scale-list [0.5,0.75,1.0,1.25,1.5,1.75,2.0] --Output ./snapshots/ --experimentID r2CityDot

```
## Dataset
I implement this on [CityScapes](https://www.cityscapes-dataset.com/) dataset.
## Thanks to the Third Party Libs
https://github.com/speedinghzl/CCNet
