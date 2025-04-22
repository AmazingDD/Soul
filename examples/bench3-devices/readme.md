

## Prepare the model

```
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py -T 4 -b 64 -dataset CIFAR10 -data_dir [dataset path] > ./logs/[logfile name].log
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py -T 4 -b 16 -dataset TinyImageNet -data_dir [dataset path] > ./logs/[logfile name].log
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py -T 10 -b 4 -dataset CIFAR10DVS -data_dir [dataset path] > ./logs/[logfile name].log
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py -T 16 -b 16 -dataset DVSGesture -data_dir [dataset path] > ./logs/[logfile name].log
```