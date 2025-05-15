import os
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from soul.model import *
from soul.neuron import *
from soul.utils import *

# CUDA_VISIBLE_DEVICES=1,3,4,5 python main.py --distributed   

# init all config settings
# args = parse_args()
# config = init_config(args)

config = {
    'seed': 2025,
    'log_dir': './logs',
    'data_dir': '/home/yudi/data/cifar10/',
    'save_dir': './saved_models/',
    'dataset': 'cifar10',
    'distributed': False,
    'workers': 4,
    'optimizer': 'adam',
    'learning_rate': 1e-4,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'batch_size': 64,
    'epochs': 70,

    'model': 'vgg9',
    'neuron_type': 'lif',
    'time_step': 4,
    'mlp_ratio': 1.0,
    'membrane_threshold': 1.0,
    'surrogate': 'atan',
}

# init logger
log_path = os.path.join(config['log_dir'], config['dataset'], config['model'], config['neuron_type'])
ensure_dir(log_path)
logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'))

# report configuration
for k, v in sorted(config.items()):
    logger.info(f'{k} = {v}')
logger.info('=' * 80)

# reproducibility
logger.info(f'Init seed {config["seed"]}...')
init_seed(config["seed"])

if config['distributed'] and 'RANK' not in os.environ:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    world_size = torch.cuda.device_count()
    mp.spawn(os.execv, args=(__file__, [__file__, *os.sys.argv[1:]]), nprocs=world_size)
    exit(0)

rank = int(os.environ.get('RANK', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', rank))
world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
distributed = config['distributed'] and world_size > 1

# load data
logger.info('Load data...')
train_dataset, test_dataset, config['input_channels'], config['num_classes'] = load_data(dataset_dir=config['data_dir'], dataset_type=config['dataset'], T=config['time_step'])
if distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
else:
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
test_sampler = torch.utils.data.SequentialSampler(test_dataset)

train_loader = get_loader(train_dataset, train_sampler, config)
test_loader = get_loader(test_dataset, test_sampler, config)

# load SNN model
logger.info(f'Load SNN model: {config["model"]} featured {config["neuron_type"]} neuron...')
model_map = {
    'vgg5': vgg5,
    'vgg9': vgg9,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16, 
    'vgg19': vgg19, 
}

neuron_map = {
    "lif": LIFNode,
    "plif": ParametricLIFNode,
    "clif": MultiStepCLIFNeuron,
    "glif": GatedLIFNode,
    "ilif": ILIF,
}

# TODO 去SJ化
from spikingjelly.activation_based import surrogate

surrogate_map = {
    'atan': surrogate.ATan(),
    # TODO 需要从spikingjelly或者别的地方扒一下常见的SG
}

logger.info(f'Init neuron type: {config["neuron_type"].upper()}')
# TODO 这里最后neuron传的参数肯定只能是config，各个neuron class自己在内部从config提取想要的参数才对
config['neuron'] = neuron_map[config['neuron_type']](surrogate_function=surrogate_map[config['surrogate']], v_threshold=config['membrane_threshold'])
model = model_map[config['model']](config)

if distributed:
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if distributed:
    model = DDP(model, device_ids=[local_rank])

criterion = nn.CrossEntropyLoss()
# TODO 加更多的optimizer
if config['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
elif config['optimizer'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
else:
    raise ValueError(f'Invalid optimizer setting {config["optimizer"]}...')

best_acc = 0.
for epoch in range(1, config['epochs'] + 1):
    model.train()
    if distributed:
        train_sampler.set_epoch(epoch)
    
    top1_meter, loss_meter = AverageMeter(), AverageMeter()
    for inputs, targets in tqdm(train_loader, unit='batch', ncols=80):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1_meter.update(acc1.item(), targets.numel())
        loss_meter.update(loss.item(), targets.numel())

    if not distributed or rank == 0:
        model.eval()

        top1_meter, loss_meter = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                loss = criterion(outputs, targets)

                loss_meter.update(loss.item(), targets.numel())
                top1_meter.update(acc1.item(), targets.numel())

        test_acc = top1_meter.avg
        logger.info(f"[Epoch {epoch}] Train Loss: {loss_meter.avg:.4f}, Acc: {top1_meter.avg:.2f}%; Test Loss: {loss_meter.avg:.4f}, Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            ensure_dir(config['save_dir'])

            best_acc = test_acc
            torch.save(
                model.module.state_dict() if distributed else model.state_dict(), 
                os.path.join(config['save_dir'], f'best_{config["model"]}_{config["neuron_type"]}_{config["dataset"]}_{config["seed"]}.pt')
            )
            logger.info(f'Best model saved with accuracy: {best_acc:.2f}%')

if distributed:
    dist.destroy_process_group()
