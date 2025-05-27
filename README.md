# Soul

SNN-based open source toolkit for edge computing.

Coming Soon...

## Overview

TBD

## Requirements

```
TBD
```

## How to run
### Command in Console 
- Running `Soul` with single GPU in default settings
    ```
    CUDA_VISIBLE_DEVICES=[GPU ID] python run_soul.py
    ```

- Running `Soul` with multiple GPUs in default settings
    ```
    CUDA_VISIBLE_DEVICES=[GPU ID1],[GPU ID2],... torchrun --nproc_per_node=[Number of used GPU] run_soul.py
    ```

### Documentation

TBD


## TODO List

- [ ] 2025.5.17-2025.5.25 解除dvs数据处理对SJ的依赖
- [ ] 2025.5.17-2025.5.25 functional reset_net抽取，去SJ化
- [ ] 2025.5.17-2025.5.25 神经元输入解耦，对齐config作为输入
- [ ] 2025.5.17-2025.5.25 neuron进行标准化、去SJ化，以base为父类进行继承开发
- [x] 2025.5.18-2025.5.19 mobicom motivation ppt
- [ ] 2025.5.20-2025.5.31 ACM computing survey V1 shoule be done
- [x] 2025.5.22-2025.5.23 Edge energy monitor include
- [x] 2025.5.22-2025.5.25 wentao FLOPs/SOPs include
- [x] 2025.5.18-2025.5.25 #Param and model size for memory footprint
- [ ] 时间待定 @wentao monitor加上对nn.linear的统计
 

## Cite

```
TBD
```
