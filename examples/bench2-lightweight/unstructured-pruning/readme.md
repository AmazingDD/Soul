## Prepare the sparsified model

```
python train.py -data_dir=[dataset path] -gpu=[GPU ID] -dataset=CIFAR10 -model=SpikingVGG9 -b=64 -T=4 -thr=0.01 
```

```
python train.py -data_dir=[dataset path] -gpu=[GPU ID] -dataset=DVSGesture -model=SewResNet18 -b=32 -T=16 -thr=0.01 
```

## Testing

```
python test.py
```