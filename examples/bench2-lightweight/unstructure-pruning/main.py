import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import functional

from utils import *
from model import SpikingVGG9, SEWResNet18
from model.layers import Mask, ConvBlock

model_map = {
    "SpikingVGG9": SpikingVGG9,
    "SEWResNet18": SEWResNet18
}

def parse_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epoch-search', default=150, type=int)
    parser.add_argument('--epoch-finetune', default=50, type=int,
                        help='when to fine tune, -1 means will not fine tune')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('-T', default=4, type=int, help='simulation steps')
    parser.add_argument('-model', default='SpikingVGG9', help='model type')
    parser.add_argument('-dataset', default='CIFAR10', help='dataset type')
    parser.add_argument('-gpu', default=0, help='device')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--output-dir', default='./neuron_prune_logs/')
    # penalty term
    parser.add_argument('-lamda', '--penalty-lmbda', type=float, default=1e-11)
    # mask init
    parser.add_argument(
        '--mask-init-factor', type=float, nargs='+', default=[0, 0, 0, 0],
        help='--mask-init-factor <weights mean> <neurons mean> <weights std> <neurons std>')

    parser.add_argument('--search-lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--finetune-lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--prune-lr', type=float, help='initial learning rate of pruning')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay (default: 0)')

    args = parser.parse_args()
    return args

def load_data(dataset_dir, dataset_type, T):
    if dataset_type == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), 
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), 
            train=False,
            download=True)
    elif dataset_type == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]]), 
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]]), 
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False, download=True)
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        transform_train = DVStransform(
            transform=transforms.Compose([transforms.Resize(size=(48, 48), antialias=True)])
        )
        transform_test = DVStransform(
            transform=transforms.Resize(size=(48, 48), antialias=True)
        )

        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        train_dataset = DatasetSplitter(dataset, 0.9, True) 
        test_dataset = DatasetSplitter(dataset, 0.1, False)
    elif dataset_type == 'DVSGesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        transform_train = DVStransform(
            transform=transforms.Compose([transforms.Resize(size=(64, 64), antialias=True)])
        )
        transform_test = DVStransform(
            transform=transforms.Resize(size=(64, 64), antialias=True)
        )

        train_dataset = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T, split_by='number')
        test_dataset = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T, split_by='number')

    elif dataset_type == 'TinyImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize, ])
        transform_test = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ])

        train_dataset = torchvision.datasets.ImageFolder(traindir)
        test_dataset = torchvision.datasets.ImageFolder(valdir)

    else:
        raise ValueError(dataset_type)
    
    dataset_train = DatasetWarpper(train_dataset, transform_train)
    dataset_test = DatasetWarpper(test_dataset, transform_test)

    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler

def train_one_epoch(model, criterion, penalty_term, optimizer_train, optimizer_prune, data_loader_train, device, prune=False):
    model.train()
    acc = 0.
    total_loss = 0.
    num_samples = 0
    set_pruning_mode(model, prune)
    for image, target in tqdm(data_loader_train, unit='batch'):
        model.zero_grad()
        image, target = image.float().to(device), target.to(device)

        output = model(image)
        loss = criterion(output, target)

        if prune:
            loss = loss + penalty_term()

        loss.backward()
        if prune:
            optimizer_prune.step()
        optimizer_train.step()

        num_samples += target.numel()
        total_loss += loss.item() * target.numel()
        acc += (output.argmax(1) == target).float().sum().item()

        functional.reset_net(model)

    acc /= num_samples
    total_loss /= num_samples

    return total_loss, acc * 100

def evaluate(model, criterion, data_loader, device, prune):
    num_samples = 0
    total_loss = 0.
    acc = 0.
    model.eval()
    set_pruning_mode(model, prune)
    with torch.no_grad():
        for image, target in data_loader:
            image = image.float().to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(image)
            loss = criterion(output, target)

            num_samples += target.numel()
            total_loss += loss.item() * target.numel()
            acc += (output.argmax(1) == target).float().sum().item()

            functional.reset_net(model)

    acc /= num_samples
    total_loss /= num_samples 

    return total_loss, acc * 100

def test(model, data_loader_test, args, device):
    safe_makedirs(os.path.join(args.output_dir, 'test'))

    set_pruning_mode(model, False)
    mon = SOPMonitor(model)

    logger.info('[Sparsity]')
    conn, total = model.connects()
    logger.info(f'Connections: left: {conn:.2e}, total: {total:.2e}, connectivity {100 * conn / total:.2f}%')

    neuron_left, neuron_total = left_neurons(model)
    weight_left, weight_total = left_weights(model)

    logger.info(f'Neurons: left: {neuron_left:.2e}, total: {neuron_total:.2e}, percentage: {(neuron_left + 1e-10) / (neuron_total + 1e-10) * 100:.2f}%')
    logger.info(f'Weights: left: {weight_left:.2e}, total: {weight_total:.2e}, percentage: {(weight_left + 1e-10) / (weight_total + 1e-10) * 100:.2f}%')

    logger.info('[Efficiency]')

    model.eval()
    mon.enable()
    logger.debug('Test start')

    num_samples = 0
    acc = 0.
    with torch.no_grad():
        for image, target in enumerate(data_loader_test):
            image, target = image.to(device), target.to(device)
            output = model(image)
            functional.reset_net(model)

            acc += (output.argmax(1) == target).float().sum().item()
            num_samples += target.numel()

    logger.info(f'Acc@1: {acc / num_samples * 100:.2f}')

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        sop = torch.cat(sublist).mean().item()
        sops = sops + sop

    sops /= (1000**3)
    sops = sops / args.batch_size
    logger.info(f'Avg SOPs: {sops:.5f} G, Energy Cost: {0.9 * sops:.5f} mJ.')

if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    logger.info(str(args))

    dataset_type = args.dataset
    if dataset_type == 'CIFAR10':
        num_classes = 10
        input_shape = (3, 32, 32)
    if dataset_type == 'CIFAR10DVS':
        num_classes = 10
        input_shape = (2, 48, 48)
    elif dataset_type == 'DVSGesture':
        num_classes = 11
        input_shape = (2, 64, 64)
    elif dataset_type == 'CIFAR100':
        num_classes = 100
        input_shape = (3, 32, 32)
    elif dataset_type == 'TinyImageNet':
        num_classes = 200
        input_shape = (3, 224, 224)

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(
        args.data_path, dataset_type, args.T)
    logger.info(f'dataset_train: {len(dataset_train)}, dataset_test: {len(dataset_test)}')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        sampler=train_sampler, 
        num_workers=args.workers,
        pin_memory=True, 
        drop_last=True)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        sampler=test_sampler, 
        num_workers=args.workers,
        pin_memory=True, 
        drop_last=False)

    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    model = model_map[args.model_type](num_classes=num_classes, T=args.T, input_shape=input_shape)
    logger.info(model)

    model.to(device)

    param_without_masks = list(model.parameters())

    optimizer_train = torch.optim.Adam(
        param_without_masks, 
        lr=args.search_lr,
        betas=(0.9, 0.999), 
        weight_decay=args.weight_decay)
    
    # init mask
    set_pruning_mode(model, True)
    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR100':
        inputs = torch.rand(1, 3, 32, 32).to(device)
    elif dataset_type == 'CIFAR10DVS':
        inputs = torch.rand(1, 1, 2, 48, 48).to(device)
    elif dataset_type == 'DVSGesture':
        inputs = torch.rand(1, 1, 2, 64, 64).to(device)
    elif dataset_type == 'ImageNet':
        inputs = torch.rand(1, 3, 224, 224).to(device)
    _ = model(inputs)

    masks = init_mask(model, *args.mask_init_factor)
    set_pruning_mode(model, False)
    functional.reset_net(model)

    if args.prune_lr is None:
        args.prune_lr = args.search_lr
    optimizer_prune = torch.optim.Adam(
        masks, lr=args.prune_lr, betas=(0.9, 0.999), weight_decay=args.prune_weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    penalty_term = PenaltyTerm(model, args.penalty_lmbda)

    # lr scheduler
    lr_scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_train, T_max=args.epoch_search)
    lr_scheduler_prune = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_prune, T_max=args.epoch_search + args.epoch_finetune)
    
    start_epoch = 0
    max_acc1 = 0

    logger.info("Search start")
    for epoch in range(start_epoch, args.epoch_search):
        logger.info(f'Epoch [{epoch}] Start, lr {optimizer_train.param_groups[0]["lr"]:.6f}')

        with Timer(' Train', logger):
            logger.debug('[Training]')
            train_loss, train_acc1 = train_one_epoch(
                model, criterion, penalty_term, optimizer_train, optimizer_prune, data_loader_train, logger, epoch, device, True)
            
            lr_scheduler_train.step()
            lr_scheduler_prune.step()

        for n, m in model.named_modules():
            if isinstance(m, Mask):
                if m.mask_value is not None:
                    logger.debug(f' {n}: {m.mask().mean() * 100:.3}%')

        with Timer(' Test', logger):
            logger.debug('[Test with continuous mask]')
            test_loss_c, test_acc1_c = evaluate(model, criterion, data_loader_test, device, True)
            logger.debug('[Test with binary mask]')
            test_loss_s, test_acc1_s = evaluate(model, criterion, data_loader_test, device, False)

        logger.info(f'Epoch {epoch}: test (continuous mask) acc: {test_acc1_c:.2f}%, test (binary mask) acc: {test_acc1_s:.2f}%')


        if max_acc1 < test_acc1_c:
            max_acc1 = test_acc1_c
            torch.save(model.state_dict(), f'./saved_sparsified_models/best_neuronprune_{args.model}_{args.dataset}.pth')
            logger.info(f'Best model saved with accuracy: {test_acc1_c:.2f}%')

        set_pruning_mode(model, True)
        n_l, n_t = left_neurons(model)
        w_l, w_t = left_weights(model)
        c, t = model.connects()
        neu, wei = 100 * (n_l + 1e-10) / (n_t + 1e-10), 100 * (w_l + 1e-10) / (w_t + 1e-10)
        conn = 100 * (c + 1e-10) / (t + 1e-10)

        logger.info(f' left neurons: {neu:.2f}%, left weights: {wei:.2f}%, connectivity: {conn:.2f}%')

    logger.info('Search finish.')

    # finetune
    if args.finetune_lr is None:
        args.finetune_lr = args.search_lr  
    for param_group in optimizer_train.param_groups:
        param_group['lr'] = args.finetune_lr

    lr_scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_train, T_max=args.epoch_finetune)

    start_epoch = 0
    logger.info('Finetune start')
    for epoch in range(start_epoch, args.epoch_finetune):
        logger.info(f'Epoch [{epoch}] Start, lr {optimizer_train.param_groups[0]["lr"]:.6f}')

        with Timer(' Train', logger):
            logger.debug('[Training]')
            train_loss, train_acc1 = train_one_epoch(
                model,
                criterion,
                None,
                optimizer_train, 
                None, 
                data_loader_train, 
                device, 
                prune=False
            )
            
            lr_scheduler_train.step()

        with Timer(' Test', logger):
            logger.debug('[Test]')
            test_loss, test_acc1 = evaluate(model, criterion, data_loader_test, device, False)

        logger.info(f' Test Acc@1: {test_acc1:.2f}')

        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            torch.save(model.state_dict(), f'./saved_sparsified_models/best_neuronprune_{args.model}_{args.dataset}.pth')
            logger.info(f'Best model saved with accuracy: {test_acc1:.2f}%')

    logger.info('Finetune finish.')


    # test
    del model

    model = model_map[args.model_type](num_classes=num_classes, T=args.T, input_shape=input_shape)
    model.to(device)

    # init mask
    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR100':
        inputs = torch.rand(1, 3, 32, 32).to(device)
    elif dataset_type == 'CIFAR10DVS':
        inputs = torch.rand(1, 1, 2, 48, 48).to(device)
    elif dataset_type == 'DVSGesture':
        inputs = torch.rand(1, 1, 2, 64, 64).to(device)
    elif dataset_type == 'ImageNet':
        inputs = torch.rand(1, 3, 224, 224).to(device)
    _ = model(inputs)

    masks = init_mask(model, 1, 1, 0, 0)
    functional.reset_net(model)

    ckpt = torch.load(f'./saved_sparsified_models/best_neuronprune_{args.model}_{args.dataset}.pth', map_location='cpu')
    model.load_state_dict(ckpt)

    del test_sampler, data_loader_test

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        sampler=test_sampler, 
        num_workers=args.workers,
        pin_memory=False, 
        drop_last=False
    )

    logger.info('Test start')
    test(model, data_loader_test, args, device)
    logger.info('All Done.')
