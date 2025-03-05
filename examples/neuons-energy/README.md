# Comparison of different neuron models for Edge SNNs

Command List
```
python test.py -T=4 -dataset=cifar10 -neuron=Izhikevich -gpu=0
python test.py -T=4 -dataset=cifar10 -neuron=LIF -gpu=0
python test.py -T=4 -dataset=cifar10 -neuron=PLIF -gpu=0
python test.py -T=4 -dataset=mnist -neuron=Izhikevich -gpu=0
python test.py -T=4 -dataset=mnist -neuron=LIF -gpu=0
python test.py -T=4 -dataset=mnist -neuron=PLIF -gpu=0
python test.py -T=10 -dataset=nmnist -neuron=Izhikevich -gpu=0
python test.py -T=10 -dataset=nmnist -neuron=LIF -gpu=0
python test.py -T=10 -dataset=nmnist -neuron=PLIF -gpu=0
```