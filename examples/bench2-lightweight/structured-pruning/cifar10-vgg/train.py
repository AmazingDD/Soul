import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import torch.distributed.optim
import argparse
from torchvision import datasets, transforms
from spikingjelly.clock_driven import functional
import utils
from tqdm import tqdm
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.optim.lr_scheduler import StepLR
_seed_ = 2020
import random
import torch.optim as optim
from tensorboardX import SummaryWriter
from spikingjelly.clock_driven.monitor import Monitor
from snnvgg import *
#from spiking_resnet_p import *
#from snnwrn import *
#from snnwrn_p import *
from torch.autograd import Variable
print('4')
writer = SummaryWriter('./')
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np

from maskk import PruningLayer, PruningNetworkManager
np.random.seed(_seed_)

import numpy as np

np.random.seed(_seed_)
writer = SummaryWriter('./')







import csv
l1_lambda=3e-5
lamda=5e-5
def l1_regularization(model, l1_alpha):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
        
            module.weight.grad.data.add_(l1_alpha * torch.sign(module.weight.data))

def train_one_epoch(model,manager, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    #i=0
    #if i!=0:
        

    for image, target in tqdm(data_loader):

        

        start_time = time.time()
        image, target = image.to(device), target.to(device)
        # with torch.autograd.detect_anomaly():
        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:

            output = model(image)
            loss = criterion(output, target)
            '''re=0
            for param in model.parameters():
                re+=torch.sum(abs(param))
            loss=loss1+lamda*re'''
            
          
        

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            #manager.prints()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            #manager.prints()
            
            l1_regularization(model, l1_lambda)
            


           

            optimizer.step()
            #manager.save_csv()
    
        #manager.prints()
       
        
    
        
        

        functional.reset_net(model)
        #i=i+1
        #manager.update_masks(model)
        #manager.do_masks(model)
        #manager.compute_prune()
        #manager.update_thres(0.2,0.1,0.2)
        i=0
        if i!=0:
            manager.do_masks(model)
            i+=1

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        #print(acc1)
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    
    #manager.update_thres(0.4,0.1,0.4)
    #manager.update_masks()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(acc1_s)
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg






def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    correct = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            #pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        #correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    #torch.save(model.state_dict(), "./vggyl.pt")
    #torch.save(model, "./vggyl.pth")
    #print('saved')
    return loss, acc1, acc5


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


tr = datasets.CIFAR10('./', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomCrop(32, padding=4),
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                      ]))

te = datasets.CIFAR10('./', train=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                      ]))
print(f'dataset_train:{tr.__len__()}, dataset_test:{te.__len__()}')

train_loader = torch.utils.data.DataLoader(
    tr,
    batch_size=64, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    te,
    batch_size=64, shuffle=True, drop_last=True)




'''mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])

cifar100_training = torchvision.datasets.CIFAR100(root='./',
                                                  train=True, download=True,
                                                  transform=transform_train)
train_loader = torch.utils.data.DataLoader(cifar100_training, batch_size=64, shuffle=True, drop_last=True)

cifar100_testing = torchvision.datasets.CIFAR100(root='./',
                                                 train=False, download=True,
                                                 transform=transform_test)
test_loader  = torch.utils.data.DataLoader(cifar100_testing, batch_size=64, shuffle=False, drop_last=True)
trainnum = len(train_loader)
print(trainnum)
totalnum = trainnum + len(test_loader)
print(totalnum)'''




def main(args):
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)
    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_lr{args.lr}_T{args.T}')

    if args.zero_init_residual:
        output_dir += '_zi'
    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    output_dir += f'_coslr{args.cos_lr_T}'

    if args.adam:
        output_dir += '_adam'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if output_dir:
        utils.mkdir(output_dir)

    device = torch.device(args.device)

    '''tr = datasets.CIFAR10('./', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(32, padding=4),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                          ]))

    te = datasets.CIFAR10('./', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                          ]))
    print(f'dataset_train:{tr.__len__()}, dataset_test:{te.__len__()}')

    train_loader = torch.utils.data.DataLoader(
        tr,
        batch_size=64, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        te,
        batch_size=64, shuffle=True, drop_last=True)'''
    '''mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar100_training = torchvision.datasets.CIFAR100(root='./',
                                                      train=True, download=True,
                                                      transform=transform_train)
    train_loader = torch.utils.data.DataLoader(cifar100_training, batch_size=64, shuffle=True, drop_last=True)

    cifar100_testing = torchvision.datasets.CIFAR100(root='./',
                                                     train=False, download=True,
                                                     transform=transform_test)
    test_loader  = torch.utils.data.DataLoader(cifar100_testing, batch_size=64, shuffle=False, drop_last=True)
    trainnum = len(train_loader).int()
    print(trainnum)
    totalnum = trainnum + len(test_loader).int()
    print(totalnum)'''








    print("Creating model")
    #model=SNNWideResNet_p().to(device)  #with prune
    #model=SNNWideResNet().to(device)    #without prune
    #model=spiking_resnet18().to(device)   #without prune
    model=snnvgg16_bn().to(device)   #with prune
    mymanager = PruningNetworkManager(model)

    '''checkpoint=torch.load('pruned.pth.tar')

    model = snnvgg16_bn(cfg=checkpoint['cfg']).to(device)
    print(model)
    model.load_state_dict(checkpoint['state_dict'])'''

    print('model')

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print('abcdef12346')
    criterion = nn.CrossEntropyLoss()
    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
              momentum=0.9, weight_decay=args.weight_decay)
    print('wwww')
    '''if args.adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)'''

    if args.amp:
        scaler = amp.GradScaler()
    else:
        print('n')
        scaler = None

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)
    lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        print('a')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

        max_test_acc1 = checkpoint['max_test_acc1']
        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()
    writer.flush()
    # checkpoint = torch.load('./vggyl.pt')
    # model.load_state_dict(checkpoint)
    # print('Model loaded.')
    #mon = my(model, device='cpu', backend='torch')
    for epoch in range(300):
        print(epoch)
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mymanager.training()
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, mymanager,criterion, optimizer, train_loader, device, epoch,
                                                             args.print_freq, scaler)
        '''if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)'''
        lr_scheduler.step()
        print(train_acc1)
        
        #mymanager.compute_prune2()
        mymanager.evaling()

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, test_loader, device=device, header='Test:')
        
        mymanager.update_masks(model,0.8,0.1)
        mymanager.do_masks(model)
        mymanager.compute_prune()
        #mymanager.update_thres(0.2,0.1,0.2)
        mymanager.save_csv()
        mymanager.reset_zeros()
        
        

        #mymanager.update_masks(0.4)
        #mymanager.compute_prune()
        #mymanager.update_thres(0.4,0.1,0.4)
        #mymanager.update_masks_grow(0.4)
        
        #mymanager.reset_zeros()
        
        
        writer.add_scalar('test_accuracy', test_acc1, epoch )
        if te_tb_writer is not None:
            if utils.is_main_process():
                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)
        print(test_acc1)
        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': max_test_acc1,
                'optimizer': optimizer.state_dict(),
            }, os.path.join('./', 'vgg16cifar10best.pth.tar'))
            #print('saved')
            mymanager.save_csv_max()

        print(max_test_acc1)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
              'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)




def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/home/wfang/datasets/ImageNet', help='dataset')

    parser.add_argument('--model', default='spiking_resnet18', help='model')
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=3200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')  
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight_decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')



    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam. The default optimizer is SGD.')

    parser.add_argument('--cos_lr_T', default=320, type=int,
                        help='T_max of CosineAnnealingLR.')
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    parser.add_argument('--zero_init_residual', action='store_true', help='zero init all residual blocks')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

'''

python m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path /raid/wfang/imagenet

python train.py --cos_lr_T 320 --model spiking_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 --data-path /raid/wfang/imagenet --device cuda:0 --zero_init_residual



'''
