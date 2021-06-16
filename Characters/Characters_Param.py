import copy
import random

import time
import numpy as np
import torch
from torch import nn

from Utils.Helper import remote_method, get_accuracy, remote_method_async, load_model, DataAggregation
from Utils.Dataloader import partition_dataset

from loguru import logger
import wandb


wandb_enable = False


class ParameterServer(nn.Module):
    def __init__(self, gpu, world_size, dataset, batch_size, lr, model, max_epoch, client_epoch, seed, exp_id):
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        self.iid = False

        logger.add(f"logs/fed/{world_size}_{dataset}_{batch_size}_{lr}"
                   f"_{model}_{max_epoch}_{client_epoch}_PS{exp_id}.log")
        if wandb_enable:
            wandb.init(project="Async_FedAvg",
                       name=f"vanilla_{world_size}_{batch_size}_{max_epoch}_{client_epoch}"
                            f"_{lr}_{dataset}_{model}_{'iid' if self.iid else 'noniid'}",
                       config={
                "method": "vanilla",
                "world size": world_size,
                "dataset": dataset,
                "iid": self.iid,
                "model": model,
                "batch size": batch_size,
                "learning rate": lr,
                "momentum": 0.,
                "lambda": 0.,
                "global epoch": max_epoch,
                "client epoch": client_epoch,
                "seed": seed
            })

        self.max_epoch = max_epoch
        self.client_epoch = client_epoch
        self.curr_epoch = -1
        self.lr = lr
        self.world_size = world_size
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.model = load_model(model, class_num=100 if dataset == "cifar100" else 10).to(self.device)
        self.aggregation = [DataAggregation(r) for r in range(1, world_size)]
        self.embedding_list = []

        # linear.bias

        self.client_counter = 0
        self.fetch_time = 0.
        self.sync_time = 0.

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.)
        _, self.test_loader = partition_dataset(dataset, world_size - 1, 0, batch_size, seed, self.iid)

    def embedding_workers(self, _rref):
        print(f"Param Server add a worker reference")
        self.embedding_list.append(_rref)

    def worker_weight_update(self, rank, weight):
        item = next((x for x in self.aggregation if x.rank == rank), None)
        if item is None:
            raise Exception("PS trys to update the weight of a worker that doesn't exists")
        item.weight = copy.deepcopy(weight)
        print(f"PS updating worker {rank}'s aggregation weight to {weight}")

    def forward(self, x):
        x = x.to(self.device)
        out = self.model(x)
        return out

    def instruct_training(self):
        total_train_time = 0.
        acc_list = []

        if wandb_enable:
            wandb.watch(self.model)

        for curr_epoch in range(self.max_epoch):
            logger.info(f"PS instructs training for epoch {curr_epoch}")
            epoch_time = []
            futs = []
            for worker_rref in self.embedding_list:
                fut = remote_method_async(TrainerNet.train_locally, worker_rref)
                futs.append(fut)
            for fut in futs:
                train_time, comm_time = fut.wait()
                epoch_time.append(train_time + comm_time)
            total_train_time += max(epoch_time)
            logger.info(f"Cluster finished training for epoch {curr_epoch}, max epoch {self.max_epoch - 1}")
            logger.info(f"Epoch {curr_epoch} takes {max(epoch_time)} seconds")

            acc = get_accuracy(self.test_loader, self.model, self.device)
            logger.info(f"Accuracy: {acc}")
            acc_list.append(acc)
            if wandb_enable:
                wandb.log({"accuracy": acc}, step=curr_epoch)
                wandb.log({"training time": max(epoch_time), "total train time": total_train_time}, step=curr_epoch)

        logger.info("Training complete!")
        acc = get_accuracy(self.test_loader, self.model, self.device)
        acc_list.append(acc)
        logger.info(f"Total train time: {total_train_time}")
        logger.info(f"Best accuracy {max(acc_list)}")
        if wandb_enable:
            wandb.log({"accuracy": acc}, step=self.max_epoch - 1)
            wandb.finish()

    def distribute_state(self, k):
        for name, param in self.model.state_dict().items():
            if name == k:
                return param.data.to("cpu")

    def synchronize(self, rank, key, data):
        item = next((x for x in self.aggregation if x.rank == rank), None)
        if item is None:
            raise Exception("PS trys to update the data of a worker that doesn't exists")
        item.data[key] = copy.deepcopy(data.to(self.device))

    def sync_counter(self):
        self.client_counter += 1
        # print(f"PS received data from {self.client_counter} workers")
        total_weights = sum([item.weight for item in self.aggregation])
        workers_weight = [item.weight / total_weights for item in self.aggregation]
        if self.client_counter == self.world_size - 1:
            self.model.train()
            # print(f"PS starts to aggregate data")
            static_state_dict = copy.deepcopy(self.model.state_dict())
            for name, param in static_state_dict.items():
                if "running_var" in name or "running_mean" in name:
                    print(name, param, "\n")
                temp_data = torch.zeros(self.aggregation[0].data[name].shape).to(self.device)
                for index in range(0, self.world_size - 1):
                    if "running_var" in name or "running_mean" in name:
                        print(index, name, self.aggregation[index].data[name], "\n")
                    if "running_var" in name:
                        temp_data +=\
                            (self.aggregation[index].data[name] +
                             self.aggregation[index].data[name.replace("running_var", "running_mean")] ** 2
                             ) * workers_weight[index]
                    else:
                        temp_data += self.aggregation[index].data[name] * workers_weight[index]
                if "running_var" in name:
                    temp_data -= static_state_dict[name.replace("running_var", "running_mean")]
                static_state_dict[name] = copy.deepcopy(temp_data).to(self.device)
                if "running_var" in name or "running_mean" in name:
                    print(name, temp_data, "\n")

            self.model.load_state_dict(static_state_dict)
            # self.optimizer.step()

            # print(f"PS aggregating completed")
            for aggregation_item in self.aggregation:
                aggregation_item.clear_data()
            self.client_counter = 0
            # self.instruct_training()


class TrainerNet(nn.Module):
    def __init__(self, gpu, rank, world_size, dataset, model,
                 batch_size, lr, client_epoch, seed, exp_id):
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        self.iid = False

        logger.add(f"logs/fed/{world_size}_{dataset}_{batch_size}_{lr}"
                   f"_{exp_id}_{rank}.log")
        self.rank = rank
        self.lr = lr
        self.client_epoch = client_epoch
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.model = load_model(model, class_num=100 if dataset == "cifar100" else 10).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.)
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader, _ = partition_dataset(dataset, world_size - 1, rank - 1, batch_size, seed, self.iid)

    def embedding_param_server(self, _rref):
        # print(f"Worker {self.rank} add a parameter server reference")
        self.param_server_rref = _rref
        remote_method(ParameterServer.worker_weight_update, self.param_server_rref, self.rank,
                      len(self.train_loader.dataset))

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def train_locally(self):
        self.model.train()
        # print(f"Worker {self.rank} starts to train locally")
        # 1. fetch the latest model from the master
        fetch_time = self.fetch_state()
        global_train_time = 0.
        # 2. do local training
        for e in range(self.client_epoch):
            self.model.train()
            total_loss = 0.
            train_start = time.time()
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.model.zero_grad()
                pred = self.model(data)
                loss = self.loss_func(pred, target)
                # if i > 0 and i % 10 == 0:
                #     logger.info(f"Rank {self.rank} training interation {i} loss: {total_loss / len(self.train_loader)}")
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_end = time.time()
            global_train_time += train_end - train_start
            logger.info(f"Rank {self.rank} training time: {train_end - train_start}")
            logger.info(f"Rank {self.rank} training loss: {total_loss / len(self.train_loader)}")
            # 3. synchronizing
        sync_time = self.synchronize()
        logger.info(f"Rank {self.rank} communication time: {sync_time + fetch_time}")
        return global_train_time, sync_time + fetch_time

    def fetch_state(self):
        fetch_time = 0.
        static_state_dict = copy.deepcopy(self.model.state_dict())
        for name, param in static_state_dict.items():
            static_state_dict[name] = copy.deepcopy(remote_method(
                ParameterServer.distribute_state, self.param_server_rref, name)).to(self.device)
            fetch_time += (1.6e-06 * param.data.size().numel())
        self.model.load_state_dict(static_state_dict)
        return fetch_time

    def synchronize(self):
        sync_time = 0.
        static_state_dict = copy.deepcopy(self.model.state_dict())
        for name, param in static_state_dict.items():
            # print(f"Worker {self.rank} starts to synchronize {name}")
            remote_method(ParameterServer.synchronize, self.param_server_rref, self.rank,
                          name, copy.deepcopy(param.data).to("cpu"))
            sync_time += (1.6e-06 * param.data.size().numel())
        # synchronizing completed, sending a signal.
        # print(f"Worker {self.rank} finished synchronizing")
        remote_method(ParameterServer.sync_counter, self.param_server_rref)
        return sync_time

