import torch
from vgg import SpikingVGG9

num_classes = 10
T = 4
input_shape = (3, 32, 32)
sample_name = 'cifar10-T4-size32.pt'

model = SpikingVGG9(T=T,num_classes=num_classes,input_shape=input_shape)
sample_name = f"../samples/{sample_name}"
sample = torch.load(sample_name,device="cpu")
features = model.features(sample)