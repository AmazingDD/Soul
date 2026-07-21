
<!--
---

<p align="center">
    <img src="./assets/code-logo.jpg" width="50%" loading="lazy">
</p>

---

*“I have always been convinced that the only way to get artificial intelligence to work is to do the computation in a way similar to the human brain.”——Geoffrey Hinton*

-->

## Overview

**Edge sensing and IoT workloads operate under stringent energy and latency constraints, for which spiking neural networks (SNNs) have emerged as a promising alternative to artificial neural networks (ANNs)**. However, despite growing interest, system-level and cross-modal evidence clarifying when and why SNNs outperform ANNs remains limited. Here, we present a unified benchmarking framework and open-source toolkit spanning datasets acquired from diverse sensing modalities, including vision, audio, motion, wireless, and event-based sensors. Under a consistent training and evaluation protocol, we systematically compare a wide range of spiking neuron models and SNN architectures against size-matched lightweight ANN baselines deployed on edge devices, jointly reporting correctness, timing, complexity and energy-efficiency. By establishing a fair, rigorous, and reproducible evaluation methodology, this work provides actionable guidance for selecting SNNs in edge sensing applications and a reference benchmark to inform algorithm–software–hardware co-design.

<p align="center">
<img src="assets/overview.png" align="center" width="100%" style="margin: 0 auto">
</p>

To support the study of recent advances in neuromorphic edge sensing, we introduce SOUL-NeuSim, an open-source benchmarking toolkit for building SNN applications. SOUL-NeuSim provides a unified and efficient framework tailored for both research and edge deployment, enabling reproducible implementation of neuromorphic computing algorithms and rapid development of new models with minimal overhead. By design, SOUL-NeuSim facilitates seamless experimentation with SNNs within a comprehensive environment that bridges academic research and real-world edge intelligence.

<p align="center">
<img src="assets/workflow.png" align="center" width="100%" style="margin: 0 auto">
</p>

## Usage

You can run the library directly from the command line. For example:

- Run `SOUL-NeuSim` on a single GPU (default settings):
    ```shell
    CUDA_VISIBLE_DEVICES=[GPU-ID] python run_soul.py -dataset=[Dataset Name] -data_dir=[Dataset Directory] -T=[Number of timesteps] -m=[Model Name] -n=[Neuron Type]
    ```

- Run `SOUL-NeuSim` on multiple GPUs (default settings):
    ```shell
    CUDA_VISIBLE_DEVICES=[GPU-ID1],[GPU-ID2],... torchrun --nproc_per_node=[Number of used GPU] run_soul.py -dataset=[Dataset Name] -data_dir=[Dataset Directory] -T=[Number of timesteps] -m=[Model Name] -n=[Neuron Type]
    ```

If you want to dive deeper into all the available command-line options, configuration settings, and parameters, please check out our [complete documentation](https://soul-docs.readthedocs.io/en/latest/).

## Dataset Support

For each dataset, we provide both a **Research/Reference Link** (to a paper or dataset description) and a **Download Link** to facilitate integration with the toolkit.

<details>
  <summary><b>Vision Sensing</b></summary>
    
| Dataset | Description | Research Link | Download Link |
|:---------:|:-------------:|:--------------:|:--------------:|
| **CIFAR10/100** | Standard small-image classification benchmarks | [Paper](https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf) | [Download](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **SVHN** | Street View House Numbers dataset for digit recognition | [Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37648.pdf) | [Download](http://ufldl.stanford.edu/housenumbers/) |
| **Tiny-ImageNet**| Scaled-down version of ImageNet (200 classes, 64×64 images) | [Paper](https://ieeexplore.ieee.org/abstract/document/5206848/) | [Download](https://www.kaggle.com/c/tiny-imagenet) |
| **MNIST** | Handwritten digit recognition benchmark | - | [Download](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) |
| **Fashion-MNIST** | Drop-in replacement for MNIST with fashion items | [Paper](https://arxiv.org/abs/1708.07747) | [Download](https://github.com/zalandoresearch/fashion-mnist) |

</details>

<details>
  <summary><b>Motion Sensing</b></summary>

| Dataset | Description | Research Link | Download Link |
|:---------:|:-------------:|:--------------:|:--------------:|
| **UCI HAR** | Smartphone-sensor human activity recognition | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231215010930) | [Download](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| **HHAR** | Heterogeneity human activity recognition (phone sensors) | [Paper](https://dl.acm.org/doi/10.1145/2809695.2809718) | [Download](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) |
| **MotionSense** | iPhone/AppleWatch sensor data for multiple activities | [Paper](https://dl.acm.org/doi/10.1145/3302505.3310068) | [Download](https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset) |
| **Shoaib** | Accelerometer/gyroscope data for activity recognition | [Paper](https://www.mdpi.com/1424-8220/14/6/10146) | [Download](https://www.researchgate.net/publication/266384007_Sensors_Activity_Recognition_DataSet) |

</details>

<details>
  <summary><b>Acoustic Sensing  </b></summary>

| Dataset | Description | Research Link | Download Link |
|:---------:|:-------------:|:--------------:|:--------------:|
| **UrbanSound8K** | Urban sounds classification, 10 classes | [Paper](https://dl.acm.org/doi/10.1145/2647868.2655045) | [Download](https://urbansounddataset.weebly.com/download-urbansound8k.html) |
| **GSC** | Short spoken-words dataset for keyword recognition | [Paper](https://arxiv.org/abs/1804.03209) | [Download](https://huggingface.co/datasets/google/speech_commands) |
| **GTZAN** | Music-genre classification across 10 genres | [Paper](https://ieeexplore.ieee.org/abstract/document/1021072) | [Download](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |
| **ESC-50** | 50 classes environmental sound classification | [Paper](https://dl.acm.org/doi/abs/10.1145/2733373.2806390) | [Download](https://github.com/karoldvl/ESC-50/archive/master.zip) |

</details>

<details>
  <summary><b>Wireless Sensing  </b></summary>
 
| Dataset | Description | Research Link | Download Link |
|:---------:|:-------------:|:--------------:|:--------------:|
| **UT-HAR** | WiFi-CSI human activity recognition | [Paper](https://ieeexplore.ieee.org/document/8067693) | [Download](https://github.com/ermongroup/Wifi_Activity_Recognition?tab=readme-ov-file) |
| **NTU-HumanID** | Device-free human gait identification via WiFi | [Paper](https://ieeexplore.ieee.org/abstract/document/9726794) | [Download](https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt) |
| **BullyDetect** | Wireless sensing dataset for bullying-incident detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10734315) | [Download](http://www.sdp8.net/Dataset?id=5ab0f5fd-a678-400a-afb2-757b2d85bc68) |
| **ARIL** | WiFi-reflection human-action recognition dataset | [Paper](https://arxiv.org/pdf/1904.04964) | [Download](http://www.sdp8.net/Dataset?id=9d263468-4869-4dbb-85aa-2c63ba0a1e0f) |
| **NTU-HAR** | Wi-Fi CSI-based human activity recognition | [Paper](https://ieeexplore.ieee.org/document/9667414) | [Download](https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt) |
| **Widar 3.0** | Wi-Fi-based gesture recognition using Body-coordinate Velocity Profile (BVP) and CSI/DFS | [Paper](https://ieeexplore.ieee.org/document/9516988) | [Download](https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt) |
| **AOPHand** | A 3D mmWave radar point-cloud dataset for real-time gesture language recognition | - | [Download](https://www.kaggle.com/datasets/ahtshamzafar/tiawr6843aop-gesture-language-data) |
| **MMactvity** | 3D mmWave radar point-cloud dataset | [Paper](https://dl.acm.org/doi/pdf/10.1145/3349624.3356768) | [Download](https://github.com/nesl/RadHAR) |

</details>

<details>
  <summary><b>Neuromorphic Sensing  </b></summary>

| Dataset | Description | Research Link | Download Link |
|:---------:|:-------------:|:--------------:|:--------------:|
| **CIFAR10-DVS** | Event-based version of CIFAR-10 captured with DVS sensor | [Paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full) | [Download](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671) |
| **DVS-Gesture** | Dynamic Vision Sensor dataset for gesture recognition | [Paper](https://ieeexplore.ieee.org/document/8100264) | [Download](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794) |
| **SHD** | Event-based audio dataset for digit recognition | [Paper](https://ieeexplore.ieee.org/document/9311226) | [Download](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) |
| **SSC** | Event-based speech commands dataset | [Paper](https://ieeexplore.ieee.org/document/9311226) | [Download](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) |

</details>

<br>

> **Note:** After downloading and extracting the datasets, you only need to point the `data_dir` argument to the root directory of the dataset in your configuration (or command-line). SOUL-NeuSim will automatically process the raw data in that directory and run accordingly.

<!-- 
## Cite

If you find SOUL-NeuSim useful for your research or development, please cite the following papers:

```
@article{deng2025edge,
  title={Edge Intelligence with Spiking Neural Networks},
  author={Deng, Shuiguang and Yu, Di and Lv, Changze and Du, Xin and Jiang, Linshan and Zhao, Xiaofan and Tong, Wentao and Zheng, Xiaoqing and Fang, Weijia and Zhao, Peng and others},
  journal={arXiv preprint arXiv:2507.14069},
  year={2025}
}
```

**This code may not be used, copied, or distributed without prior written permission from the authors.**

 -->



## License

SOUL-NeuSim is released under the [Apache-2.0 License](./LICENSE). All datasets used in this project are intended for academic research purposes only.
