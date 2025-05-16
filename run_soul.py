import os
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from soul.model import *
from soul.neuron import *
from soul.utils import *


# init all config settings TODO
# args = parse_args()
# config = init_config(args)

config = {
    'seed': 2025,
    'log_dir': './logs',
    'data_dir': '~/data/cifar10/',
    'save_dir': './saved_models/',
    'dataset': 'cifar10',
    'distributed': False,
    'workers': 4,
    'optimizer': 'adam',
    'scheduler': 'cosine',
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

# activate distributed
config['is_distributed'] = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if config['is_distributed']:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    global_rank = dist.get_rank()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    global_rank = 0

# init logger
if global_rank == 0:
    log_path = os.path.join(config['log_dir'], config['dataset'], config['model'], config['neuron_type'])
    ensure_dir(log_path)
    logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'))
    logger.info(f'Distributed Training: {config["is_distributed"]}')
else:
    logger = None

# report configuration
for k, v in sorted(config.items()):
    if global_rank == 0:
        logger.info(f'{k} = {v}')

# reproducibility
if global_rank == 0:
    logger.info(f'Init seed {config["seed"]}...')
    init_seed(config["seed"])
    logger.info('=' * 50)

# load data
if global_rank == 0:
    logger.info('Load data...')
train_dataset, test_dataset, config['input_channels'], config['num_classes'] = load_data(dataset_dir=config['data_dir'], dataset_type=config['dataset'], T=config['time_step'])
if config['is_distributed']:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader, test_loader = get_loader(train_dataset, test_dataset, train_sampler, config)

# load SNN model
if global_rank == 0:
    logger.info(f'Load SNN model: {config["model"]} featured {config["neuron_type"].upper()} neuron...')
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

# TODO 这里最后neuron传的参数肯定只能是config，各个neuron class自己在内部从config提取想要的参数才对
config['neuron'] = neuron_map[config['neuron_type']](surrogate_function=surrogate_map[config['surrogate']], v_threshold=config['membrane_threshold'])
model = model_map[config['model']](config)
model.to(device)

if config['is_distributed']:
    model = DDP(model, device_ids=[local_rank])

criterion = nn.CrossEntropyLoss()
# init optimzer
if config['optimizer'].lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
elif config['optimizer'].lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
elif config['optimizer'].lower() == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
elif config['optimizer'].lower() == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
else:
    if global_rank == 0:
        logger.warning(f"Received unrecognized optimizer {config['optimizer']}, set default Adam optimizer")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

# init scheduler
if config['scheduler'].lower() == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
elif config['scheduler'].lower() == 'linear':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config["epochs"] * 0.25), gamma=0.1)
elif config['scheduler'].lower() == 'warmup':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(config["epochs"] * 0.1), T_mult=2)
else:
    if global_rank == 0:
        logger.warning(f"Received unrecognized scheduler {config['scheduler']}, set default ConsineAnnealing Scheduler")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

best_acc = 0.
for epoch in range(1, config['epochs'] + 1):
    model.train()
    if config['is_distributed']:
        train_sampler.set_epoch(epoch)
    
    top1_meter, loss_meter = AverageMeter(), AverageMeter()
    # customize progress bar for train loader
    loader = tqdm(train_loader, unit='batch', ncols=80) if global_rank == 0 else train_loader
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1_meter.update(acc1.item(), targets.numel())
        loss_meter.update(loss.item(), targets.numel())

    if not config['is_distributed'] or dist.get_rank() == 0:
        model.eval()

        top1_meter, loss_meter = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                loss = criterion(outputs, targets)

                loss_meter.update(loss.item(), targets.numel())
                top1_meter.update(acc1.item(), targets.numel())

        test_acc = top1_meter.avg
        if global_rank == 0:
            logger.info(f"[Epoch {epoch}] Train Loss: {loss_meter.avg:.4f}, Acc: {top1_meter.avg:.2f}%; Test Loss: {loss_meter.avg:.4f}, Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            ensure_dir(config['save_dir'])

            best_acc = test_acc
            if global_rank == 0:
                logger.info(f'Best model saved with accuracy: {best_acc:.2f}%')
            torch.save(
                model.module.state_dict() if config['is_distributed'] else model.state_dict(), 
                os.path.join(config['save_dir'], f'best_{config["model"]}_{config["neuron_type"]}_{config["dataset"]}_{config["seed"]}.pt')
            )

    scheduler.step()

if config['is_distributed']:
    dist.destroy_process_group()
