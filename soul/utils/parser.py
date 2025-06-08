import os
import re
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='arguments for soul')
    # Basic Settings
    parser.add_argument(
        "--workers", 
        "-j", 
        type=int, 
        default=4, 
        help="number of workers"
    )
    parser.add_argument(
        "--seed", 
        "-seed", 
        type=int, 
        default=2025, 
        help="random seed"
    )
    parser.add_argument(
        "--log_dir", 
        "-log_dir", 
        type=str, 
        default='./logs/', 
        help="save path for logs"
    )
    parser.add_argument(
        "--data_dir", 
        "-data_dir", 
        type=str, 
        default='~/data/', 
        help="path for input datasets"
    )
    parser.add_argument(
        "--model_dir", 
        "-model_dir", 
        type=str, 
        default='./saved_models/', 
        help="path for saved models"
    )
    parser.add_argument(
        "--epochs", 
        "-epochs", 
        type=int, 
        default=150, 
        help="number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", 
        "-b", 
        type=int, 
        default=128, 
        help="batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        "-lr", 
        type=float, 
        default=1e-4, 
        help="learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        "-wd", 
        type=float, 
        default=0.0, 
        help="weight decay value"
    )
    parser.add_argument(
        "--momentum", 
        "-momentum", 
        type=float, 
        default=0.9, 
        help="inertia coefficient for optimizer"
    )
    parser.add_argument(
        "--optimizer", 
        "-optimizer", 
        type=str, 
        default='adam', 
        help="optimizer name, [optional] adam, sgd, adamw, rmsprop"
    )
    parser.add_argument(
        "--scheduler", 
        "-scheduler", 
        type=str, 
        default='cosine', 
        help="scheduler name, [optional] cosine, linear, warmup"
    )
    # Specific Settings
    parser.add_argument(
        "--dataset_name", 
        "-dataset", 
        type=str, 
        default='cifar10', 
        help="datset name"
    )
    parser.add_argument(
        "--model", 
        "-m", 
        type=str, 
        default='SpikingVGG9', 
        help="model name"
    )
    parser.add_argument(
        "--neuron_type", 
        "-n", 
        type=str, 
        default='LIF', 
        help="LIF neuron type"
    )
    parser.add_argument(
        "--time_step", 
        "-T", 
        type=int, 
        default=4, 
        help="number of time steps"
    )
    
    args = parser.parse_args()
    return args


def init_config():
    current_path = os.path.dirname(os.path.realpath(__file__))
    
    # update args
    args = parse_args()
    config = vars(args)
    
    # load basic yaml
    overall_init_file = os.path.join(current_path, "../config/basic.yaml")
    config.update(yaml.safe_load(open(overall_init_file, 'r', encoding="utf-8")))

    # load model specific yaml
    match = re.match(r'^([a-zA-Z]+)', config['model'])
    if match:
        model_cofig_name = match.group(1)
    else:
        raise NotImplementedError(f'No yaml config for model: {config["model"]}')
    target_config_file = os.path.join(current_path, f"../config/model/{model_cofig_name.lower()}.yaml")
        
    model_default_config = yaml.safe_load(open(target_config_file, 'r', encoding="utf-8"))
    config.update(model_default_config)

    # load neuron specific yaml
    target_config_file = os.path.join(current_path, f"../config/neuron/{config['neuron_type'].lower()}.yaml")
    neuron_default_config = yaml.safe_load(open(target_config_file, 'r', encoding="utf-8"))
    config.update(neuron_default_config)
    
    return config
