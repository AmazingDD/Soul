import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from soul.model import *
from soul.neuron import *
from soul.utils import *

# init all config settings
config = init_config()

# activate distributed
config['is_distributed'] = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if config['is_distributed']:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # gpu for current process
    device = torch.device("cuda", local_rank)
    # main process
    global_rank = dist.get_rank()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    global_rank = 0

# init logger
if global_rank == 0:
    log_path = os.path.join(
        config['log_dir'], 
        config['dataset_name'].lower(), 
        config['model'].lower(), 
        config['neuron_type'].lower()
    )
    ensure_dir(log_path)
    logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'), default_level=config['state'])
    logger.info(f'Distributed Training: {config["is_distributed"]}')
else:
    logger = None

# report all configuration
for k, v in sorted(config.items()):
    if global_rank == 0:
        logger.info(f'{k} = {v}')

# reproducibility
if global_rank == 0:
    logger.info(f'Reproducibility with random seed {config["seed"]}')
    init_seed(config["seed"])
    logger.info('=' * 50)

# load data
if global_rank == 0:
    logger.info('Load data...')
train_dataset, test_dataset, config['input_channels'], config['input_height'], config['input_width'], config['num_classes'] = load_data(
    dataset_dir=config['data_dir'], 
    dataset_type=config['dataset_name'], 
    T=config['time_step']
)
if config['is_distributed']:
    train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    # define the batch size per gpu, usually we define the numer of process equal to the number of used gpus
    world_size = dist.get_world_size()
    config['batch_size'] //= world_size
else:
    train_sampler = None

train_loader, test_loader = get_loader(train_dataset, test_dataset, train_sampler, config)

# load SNN model
if global_rank == 0:
    logger.info(f'Load SNN model: {config["model"]} featured {config["neuron_type"].upper()} neuron...')
model_map = {
    'spikingvgg5': SpikingVGG5, 'spikingvgg9': SpikingVGG9, 'spikingvgg11': SpikingVGG11, 'spikingvgg13': SpikingVGG13, 'spikingvgg16': SpikingVGG16, 'spikingvgg19': SpikingVGG19, 
    'sewresnet18': SEWResNet18, 'sewresnet34': SEWResNet34, 'sewresnet50': SEWResNet50,
    'spikformer2': Spikformer2, 'spikformer4': Spikformer4, 'spikformer8': Spikformer8,
}

neuron_map = {
    "lif": LIFNode,
    "plif": ParametricLIFNode,
    "clif": CLIFNode,
    "glif": GatedLIFNode,
    "ilif": ILIFNode,
    # TODO
}

surrogate_map = {
    'atan': ATan(),
    'erf': Erf(),
    'rect': Rectangular(),
    'sigmoid': FastSigmoid(),
    'quant': Quant(),
    'quant4': Quant4(),
    'rectangle': Rectangle(),
}
if global_rank == 0:
    logger.debug(f'surrogate function: {config["surrogate"]}')
config['surrogate_function'] = surrogate_map[config['surrogate']]

neuron_params = {}
optional_params = ['v_threshold', 'surrogate_function', 'v_reset', 'decay_input', 'tau', 'init_tau', 'detach_reset', 'store_v_seq', 'T', 'init_v_threshold', 'init_conduct']
for param_name in optional_params:
    if param_name in config:
        neuron_params[param_name] = config[param_name]
config['neuron'] = neuron_map[config['neuron_type'].lower()](**neuron_params)


model = model_map[config['model'].lower()](config)
logger.info('\n'+ str(model))
model.to(device)

# calculate number of parameters
if global_rank == 0:
    n_parameters = sum(p.numel() for p in model.parameters() if hasattr(p, 'requires_grad'))
    logger.info(f"Number of params for model {config['model']}: {n_parameters / 1e6:.2f} M")

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

        logger.info(f"[Epoch {epoch}] Train Loss: {loss_meter.avg:.4f}, Acc: {top1_meter.avg:.2f}%; Test Loss: {loss_meter.avg:.4f}, Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            ensure_dir(config['model_dir'])

            best_acc = test_acc
            logger.info(f'Best model saved with accuracy: {best_acc:.2f}%')
            torch.save(
                model.module.state_dict() if config['is_distributed'] else model.state_dict(), 
                os.path.join(config['model_dir'], f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pt')
            )

    scheduler.step()

# monitor max memory footprint with the best model
if not config['is_distributed'] or dist.get_rank() == 0:
    best_model_path = os.path.join(config['model_dir'], f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pt')
    logger.info(f'The size of model parameter checkpoint file: {os.path.getsize(best_model_path) / (1024 ** 2):.2f} MB')
    best_params = torch.load(
        best_model_path, 
        map_location='cpu'
    )
    if config['is_distributed']:
        model.module.load_state_dict(best_params)
    else:
        model.load_state_dict(best_params)
    logger.debug(f'current device to monitor: {device}')
    model.to(device)
    model.eval()

    new_batch_size = 1
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=new_batch_size, shuffle=False, num_workers=0)

    # calculate theoretical energy cost per sample inference
    logger.info('Counting FLOPs/SOPs for theoretical inference cost')
    from soul.utils.count_sops import ops_monitor, MODULE_SOP_DICT
    ops_monitor(model, is_sop=config['sop'])
    for inputs, _ in tqdm(test_loader, unit='batch', ncols=80):
        inputs = inputs.to(device)
        _ = model(inputs)

    total_sops = 0
    for k, v in MODULE_SOP_DICT.items():
        total_sops += v

    cost_per_op = 0.9 if config['sop'] else 4.6   # pJ on 45nm hardware
    logger.info(f"Number of {'SOPs' if config['sop'] else 'FLOPs'} for model {config['model']} inference per {new_batch_size} sample: {total_sops / 1e6:.2f} M, theoretical energy cost: {total_sops * cost_per_op / 1e9:.2f} mj")

    # evluate the maximum actual memory usage for inference
    logger.info('Monitoring maximum memory usage for inference')
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with torch.inference_mode():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)  
    end_time = time.time() 
    final_max_mem = torch.cuda.max_memory_allocated()
    
    logger.info(f"Actual GPU memory usage for inference {new_batch_size} samples per batch with {config['model']}: {final_max_mem / (1024 ** 2):.2f} MB")
    
    latency = (end_time - start_time) / len(test_loader)
    logger.info(f'Inference latency for {new_batch_size} samples per batch with {config["model"]}: {latency * 1000:.2f} ms')

# recycle all process
if config['is_distributed']:
    dist.destroy_process_group()
