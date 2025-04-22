# RQ1: Bechmarking the impact of neuron type on EdgeSNN

## How to run:

### Training:

```
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py --T 4 --num_classes 10 --epochs 100 --batch_size 64 --dataset CIFAR10 --data_dir [dataset path] --lr 5e-4 --neuron_type [LIF model] --model_type [model name] > [logfile name].log
```

### Testing in Jetson Orin AGX

```
python test.py -dataset=[dataset_name] -seed=[seed_num] -neuron_type=[neuron_type]
```