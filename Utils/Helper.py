import random, os, math
import numpy as np

import torch
import torch.distributed.rpc as rpc
import torch.nn.functional as F


def rpc_test(x):
    print(f"rpc test successfully: {x}")
    return 0


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs, timeout=0)


def remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), call_method, args=args, kwargs=kwargs, timeout=0)


def remote_method_remote(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.remote(rref.owner(), call_method, args=args, kwargs=kwargs, timeout=0)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def get_accuracy(test_loader, model, device, rnn=False):
    model.eval()
    correct_sum = 0.0
    val_loss = 0.0
    # Use GPU to evaluate if possible
    with torch.no_grad():
        if rnn:
            bptt = 35
            ntokens = 33278
            num_batches = 0
            for i in range(0, test_loader.size(0) - 1, bptt):
                num_batches += 1
                inputs, target = get_batch(test_loader, i, bptt)
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                output = output.view(-1, ntokens)
                val_loss += len(inputs) * F.nll_loss(output, target).item()

            val_loss /= (len(test_loader) - 1)
            return math.exp(val_loss)
        else:
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                out = model(data)
                pred = out.argmax(dim=1, keepdim=True)
                pred, target = pred.to(device), target.to(device)
                correct = pred.eq(target.view_as(pred)).sum().item()
                correct_sum += correct

            acc = correct_sum / len(test_loader.dataset)
            return acc


class DataAggregation(object):
    def __init__(self, rank):
        self.rank = rank
        self.weight = 0
        self.data = {}

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def clear_data(self):
        self.data = {}


class ExtraWorkLoader(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataiter = iter(dataloader)

    def retrieve(self):
        try:
            data, target = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            data, target = next(self.dataiter)
        return data, target


def adjust_learning_rate(base_lr, epoch):
    return base_lr * 0.98 ** epoch


def load_model(model_name, class_num):
    if model_name == "mnistnet":
        from Networks.MnistNet import MnistNet
        return MnistNet(class_num=class_num)
    if model_name == "resnet":
        from Networks.Resnet import ResNet18
        return ResNet18(class_num=class_num)
    if model_name == "vgg":
        from Networks.VGG import VGG
        return VGG("VGG11star", class_num=class_num)
    if model_name == "mlp":
        from Networks.MLP import MLP
        return MLP(784, 200, 10)
    if model_name == "cnncifar":
        from Networks.CNNCifar import CNNCifar
        return CNNCifar(class_num)
    if model_name == "transformer":
        from Networks.Transformer import TransformerModel
        ntokens = 33278
        emsize = 200
        nhead = 2
        nhid = 200
        nlayers = 2
        dropout = 0.2
        return TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")


if __name__ == "__main__":
    from Networks.Transformer import TransformerModel
    ntokens = 33278
    emsize = 200
    nhead = 2
    nhid = 200
    nlayers = 2
    dropout = 0.2
    count_parameters(TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout))