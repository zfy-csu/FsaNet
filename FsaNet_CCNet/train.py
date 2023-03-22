import argparse
import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import os
from tqdm import tqdm
import os.path as osp
import networks
from dataset.datasets import CSDataSet
import torch.nn as nn

from loss.criterion import CriterionDSN, CriterionOhemDSN
from engine import Engine
os.environ['CUDA_VISIBLE_DEVICES']='0'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
BATCH_SIZE = 2
DATA_NAME = 'CSDataSet'
DATA_DIRECTORY = 'dataset/cityscapes'
DATA_LIST_PATH = 'dataset/list/cityscapes/train.lst'
NUM_CLASSES = 19
IGNORE_LABEL = 255
INPUT_SIZE = '269,269'
RESTORE_FROM = 'dataset/resnet101-imagenet.pth'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005
RANDOM_SEED = 12345
SNAPSHOT_DIR = './snapshots/'
SAVE_NUM_IMAGES = 2
START_ITERS = 0
NUM_STEPS = 2500
SAVE_PRED_EVERY = 500

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-name", type=str, default=DATA_NAME,
                        help="name of dataset.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=START_ITERS,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="choose the number of workers.")
    parser.add_argument("--ohem", type=str2bool, default='False',
                        help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.7,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem-keep", type=int, default=100000,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--train-mode", type=str, default=None,
                    help="if only train attention part(att_only) or only train back part(back_only), otherwise set None") 
    parser.add_argument("--model", type=str, default='ccnet',
                        help="choose model.")
    parser.add_argument("--att-mode", type=str, default='DCTNLAttention11',
                        help="choose attention mode.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.") 
    parser.add_argument('--k', type=str, default="[0, 8, 0, 8]")
    parser.add_argument("--experimentID", type=str, default='CityDot',
                        help="use for saving model file.")
    parser.add_argument("--Output", type=str, default='./snapshots/',
                        help="Where restore model parameters from.")

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    lr = base_lr*((1-float(iter)/max_iter)**(power))
    return lr
            
def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('SynchronizedBatchNorm2d') != -1:#BatchNorm
        m.eval()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

def main():  
    """Create the model and start the training."""
    parser = get_parser()
    print("Let's use", torch.cuda.device_count(),"GPU")
    

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        print('args.data_dir', args.data_dir)
        
        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # data loader
        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)
        dataset = eval(args.data_name)(args.data_dir, args.data_list, crop_size=input_size, 
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
        train_loader, train_sampler = engine.get_train_loader(dataset)

        # config network and criterion
        if args.ohem:
            criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
        else:
            criterion = CriterionDSN() #CriterionCrossEntropy()

        seg_model = eval('networks.' + args.model + '.Seg_Model')(
            num_classes=args.num_classes, criterion=criterion,
            pretrained_model=args.restore_from, recurrence=args.recurrence,
            train_mode = args.train_mode, att_mode = args.att_mode, k = eval(args.k)
        )      
        # seg_model.init_weights()
        
        # group weight and config optimizer
        optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, seg_model.parameters()), 'lr': args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #zfy, cuda->cuda:0
        seg_model.to(device)#zfy

        model = torch.nn.DataParallel(seg_model)#engine.data_parallel(seg_model) 
        model.train()
        model.apply(set_bn_eval)#fix batchnorm

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)
            
        run = True
        global_iteration = args.start_iters
        OnceFlag = True
        while run:
            epoch = global_iteration // len(train_loader)
            if engine.distributed:
                train_sampler.set_epoch(epoch)
            
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(len(train_loader)), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)
            print('start')
            
            for idx in pbar:
                global_iteration += 1
                
                images, labels, _, _ = dataloader.next()
                images = images.cuda()
                labels = labels.long().cuda()
                
                optimizer.zero_grad()
                lr = adjust_learning_rate(optimizer, args.learning_rate, global_iteration-1, args.num_steps, args.power)
                #print('lr',lr)
                loss = model(images, labels).mean()
           
                reduce_loss = engine.all_reduce_tensor(loss)#loss.data#
                loss.backward()
                optimizer.step()
            
            
                print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                        + ' Iter{}/{}:'.format(idx + 1, len(train_loader)) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduce_loss.item()                      
                
                pbar.set_description(print_str, refresh=False)
                
                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if global_iteration % args.save_pred_every == 0 or global_iteration == args.num_steps:
                        print('taking snapshot ...')
                        torch.save(seg_model.state_dict(),osp.join(args.snapshot_dir, args.experimentID+'_'+str(global_iteration)+'.pth')) 
                    
                if global_iteration >= args.num_steps:
                    run = False
                    break    
                
                
if __name__ == '__main__':
    main()
