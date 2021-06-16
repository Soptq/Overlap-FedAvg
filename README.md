# Overlap-FedAvg
Official Pytorch implementation of "Communication-Efficient Federated Learning with Compensated Overlap-FedAvg"

> While petabytes of data are generated each day by a number of independent computing devices, only a few of them can be finally collected and used for deep learning (DL) due to the apprehension of data security and privacy leakage, thus seriously retarding the extension of DL. In such a circumstance, federated learning (FL) was proposed to perform model training by multiple clients' combined data without the dataset sharing within the cluster. Nevertheless, federated learning with periodic model averaging (FedAvg) introduced massive communication overhead as the synchronized data in each iteration is about the same size as the model, and thereby leading to a low communication efficiency. Consequently, variant proposals focusing on the communication rounds reduction and data compression were proposed to decrease the communication overhead of FL. In this paper, we propose Overlap-FedAvg, an innovative framework that loosed the chain-like constraint of federated learning and paralleled the model training phase with the model communication phase (i.e., uploading local models and downloading the global model), so that the latter phase could be totally covered by the former phase. Compared to vanilla FedAvg, Overlap-FedAvg was further developed with a hierarchical computing strategy, a data compensation mechanism, and a nesterov accelerated gradients (NAG) algorithm. In Particular, Overlap-FedAvg is orthogonal to many other compression methods so that they could be applied together to maximize the utilization of the cluster. Besides, the theoretical analysis is provided to prove the convergence of the proposed framework. Extensive experiments conducting on both image classification and natural language processing tasks with multiple models and datasets also demonstrate that the proposed framework substantially reduced the communication overhead and boosted the federated learning process.

## Quick Start

### Cloning

```
git clone https://github.com/Soptq/Overlap-FedAvg
cd Overlap-FedAvg
```

### Installation

```
pip install -r requirements.txt
```

### Dataset Preparation

```
python prepare_data.py
```

### Run simulation

Here, to simplify the demonstrating process, the provided Overlap-FedAvg demo will be ran under simulated distributed environment with 5 GPUs (GPU:0 - GPU:4) installed in 1 server, where GPU:0 is the master node (parameter server) and the other GPUs are the slave nodes. However, although simulating in only 1 server requires less efforts to setup environment, the data communicating speed is drastically accelerated, and therefore introduce biases to the algorithm evaluation. Consequenlty, it is recommended to setup a real federated learning environment to test out the efficacy of Overlap-FedAvg.

The vanilla FedAvg algorithm can be tested by:

```
python FedAvg.py --world_size 5 --lr 0.001 --batch_size 32 --max_epoch 10 --client_epoch 5 --dataset mnist --model mlp --gpus 0,1,2,3,4
```

Similarly, the Overlap-FedAvg algorithm:

```
python AsyncFedAvg.py --world_size 5 --lr 0.001 --batch_size 32 --max_epoch 10 --client_epoch 5 --dataset mnist --model mlp --gpus 0,1,2,3,4
```

## Cite

```
@article{zhou2020communication,
  title={Communication-Efficient Federated Learning with Compensated Overlap-FedAvg},
  author={Zhou, Yuhao and Qing, Ye and Lv, Jiancheng},
  journal={IEEE transactions on parallel and distributed systems},
  publisher={IEEE}
}
```

## License

```
MIT License

Copyright (c) 2020 Soptq

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
