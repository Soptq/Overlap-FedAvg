import copy
import time
import random

import torch
from torch import nn
import numpy as np

from Utils.Helper import remote_method, get_accuracy, remote_method_async, \
    DataAggregation, load_model, ExtraWorkLoader
from Utils.Dataloader import partition_dataset

from loguru import logger
import wandb


wandb_enable = True


class ParameterServer(nn.Module):
    def __init__(self, gpu, world_size, dataset, batch_size, lr,
                 mom, lambd, model, max_epoch, client_epoch, seed, exp_id,
                 early_stop_round, early_stop_metric):
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        self.iid = False

        logger.add(f"logs/asyncfed/{world_size}_{dataset}_{batch_size}_{lr}"
                   f"_{mom}_{lambd}_{model}_{max_epoch}_{client_epoch}_PS{exp_id}.log")
        if wandb_enable:
            wandb.init(project="Async_FedAvg",
                       name=f"async_{world_size}_{batch_size}_{max_epoch}_{client_epoch}_{lr}_{lambd}"
                            f"_{mom}_{dataset}_{model}_{'iid' if self.iid else 'noniid'}",
                       config={
                "method": "async",
                "world size": world_size,
                "dataset": dataset,
                "iid": self.iid,
                "model": model,
                "batch size": batch_size,
                "learning rate": lr,
                "momentum": mom,
                "lambda": lambd,
                "global epoch": max_epoch,
                "client epoch": client_epoch,
                "seed": seed,
                "mom_metho": "normal",
            })

        self.max_epoch = max_epoch * client_epoch
        self.client_epoch = client_epoch
        self.world_size = world_size
        self.mom = mom
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.model = load_model(model, class_num=100 if dataset == "cifar100" else 10).to(self.device)
        self.lr = lr
        self.lambd = lambd
        # self.learning_rate = learning_rate
        self.aggregation = [DataAggregation(r) for r in range(1, world_size)]
        self.embedding_list = []

        self.dyn_task = np.array([0. for _ in range(self.world_size - 1)])
        self.dyn_timer = np.array([0. for _ in range(self.world_size - 1)])

        self.client_counter = 0
        self.wtminus1 = {}
        self.mom_buffer = {}
        self.gminus1 = {}
        self.broadcast_fut_all = None
        self.cluster_is_ready = True

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.)
        _, self.test_loader = partition_dataset(dataset, world_size - 1, 0, batch_size, seed, iid=self.iid)

        self.acc_list = []
        self.early_stop_round = early_stop_round
        self.early_stop_metric = early_stop_metric

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
        # pre-initialization
        for name, param in self.model.named_parameters():
            self.wtminus1[name] = copy.deepcopy(param.data)
            self.mom_buffer[name] = torch.zeros(param.data.shape).to(self.device)

        if wandb_enable:
            wandb.watch(self.model)

        for curr_epoch in range(self.max_epoch):
            if curr_epoch == int(self.early_stop_round) * self.client_epoch \
                    and len(self.acc_list) != 0 \
                    and max(self.acc_list[-5:]) < self.early_stop_metric:
                print(f"{curr_epoch // self.client_epoch - 1}: {max(self.acc_list[-5:])}, "
                      f"This experiment seems to be bad, early stopping")
                self.max_epoch = curr_epoch
                break

            if curr_epoch % self.client_epoch == 0:
                logger.info(f"PS instructs training for epoch {curr_epoch // self.client_epoch}")
                epoch_time = 0.
            cluster_is_ready = self.cluster_is_ready & (curr_epoch > 0) & ((curr_epoch + 1) % self.client_epoch != 0)
            self.cluster_is_ready = (not cluster_is_ready) | (curr_epoch == 0)
            if curr_epoch > 0 and curr_epoch % self.client_epoch == 0:
                # nestorv_reversed_model = copy.deepcopy(self.model).to(self.device)
                # for name, param in nestorv_reversed_model.named_parameters():
                #     param.data = param.data + self.lr * self.mom * self.mom_buffer[name]
                acc = get_accuracy(self.test_loader, self.model, self.device)
                # del nestorv_reversed_model
                logger.info(f"Accuracy: {acc}")
                self.acc_list.append(acc)
                if wandb_enable:
                    wandb.log({"accuracy": acc}, step=curr_epoch // self.client_epoch - 1)

            # epoch_start_time = time.time()
            futs = []
            train_time_list = [0]
            for worker_rref in self.embedding_list:
                fut = remote_method_async(TrainerNet.train_locally, worker_rref,
                                          curr_epoch, cluster_is_ready)
                futs.append(fut)

            for fut in futs:
                w_rank, train_time, comm_time = fut.wait()
                self.dyn_timer[w_rank - 1] += train_time
                train_time_list.append(train_time)
            # epoch_end_time = time.time()
            epoch_time += max(train_time_list)
            if (curr_epoch + 1) % self.client_epoch == 0:
                total_train_time += epoch_time
                logger.info(f"Cluster finished training for epoch {curr_epoch // self.client_epoch}, "
                            f"max epoch {self.max_epoch // self.client_epoch - 1}")
                logger.info(f"Epoch {curr_epoch // self.client_epoch} takes {epoch_time} seconds")
                if wandb_enable:
                    wandb.log({"training time": epoch_time, "total train time": total_train_time}, step=curr_epoch // self.client_epoch)

        self.cluster_is_ready = False
        for worker_rref in self.embedding_list:
            remote_method_async(TrainerNet.synchronize, worker_rref)
        while not self.cluster_is_ready:
            print(f"PS waiting for final synchronization")
            time.sleep(1)
        logger.info("Training complete!")

        # nestorv_reversed_model = copy.deepcopy(self.model).to(self.device)
        # for name, param in nestorv_reversed_model.named_parameters():
        #     param.data = param.data + self.lr * self.mom * self.mom_buffer[name]
        acc = get_accuracy(self.test_loader, self.model, self.device)
        # del nestorv_reversed_model
        logger.info(f"Accuracy: {acc}")
        self.acc_list.append(acc)
        logger.info(f"Total train time: {total_train_time}")
        logger.info(f"Best accuracy {max(self.acc_list)}")
        if wandb_enable:
            wandb.log({"accuracy": acc}, step=self.max_epoch // self.client_epoch - 1)
            wandb.finish()

    # def retrieve_cluster_state(self):
    def distribute_state(self, k):
        for name, param in self.model.named_parameters():
            if name == k:
                return param.data.to("cpu")

    def synchronize(self, rank, key, data):
        # print(f"PS starts to synchronize {key} from Worker {rank}")
        # time.sleep(1.6e-06 * data.size().numel())
        item = next((x for x in self.aggregation if x.rank == rank), None)
        if item is None:
            raise Exception("PS trys to update the data of a worker that doesn't exists")
        # pri nt(f"PS finished synchronizing")
        item.data[key] = copy.deepcopy(data.to(self.device))

    def set_cluster_ready(self):
        # print("PS finished broadcasting model")
        self.cluster_is_ready = True

    def broadcast_state(self):
        # print("PS starts to broadcast model")
        broadcast_futs = []
        v_train = (1 + self.dyn_task) / self.dyn_timer
        self.dyn_task = 0.9 * v_train / v_train.min() - 0.9
        self.dyn_timer *= 0.
        for w_rank, worker_rref in enumerate(self.embedding_list):
            extra_work_fut = remote_method_async(TrainerNet.recv_extra_work, worker_rref, self.dyn_task[w_rank])
            broadcast_futs.append(extra_work_fut)
            for name, param in self.model.named_parameters():
                broad_fut = remote_method_async(TrainerNet.recv_state, worker_rref, name,
                                                param.data.to("cpu"))
                broadcast_futs.append(broad_fut)
        self.broadcast_fut_all = torch.futures.collect_all(broadcast_futs)
        self.broadcast_fut_all.then(lambda x: self.set_cluster_ready())

    def model_update(self):
        self.model.train()
        # print(f"PS starts to update model")
        total_weights = sum([item.weight for item in self.aggregation])
        workers_weight = [item.weight / total_weights for item in self.aggregation]
        for name, param in self.model.named_parameters():
            aggregation_weights = torch.zeros(self.aggregation[0].data[name].shape).to(self.device)
            compensation = torch.zeros(aggregation_weights.shape).to(self.device)
            for index in range(0, self.world_size - 1):
                worker_grad = copy.deepcopy(self.wtminus1[name] - self.aggregation[index].data[name])\
                    .to(self.device)
                compensation += copy.deepcopy(workers_weight[index] * worker_grad * worker_grad).to(self.device)
                aggregation_weights += copy.deepcopy(workers_weight[index] * self.aggregation[index].data[name])\
                    .to(self.device)
            # d_grad_mul_lr = (self.wtminus1[k] - aggregation_weights) + \
            #                 (self.lambd * (state_dict[k] - self.wtminus1[k]) * compensation) / self.lr

            ps_grad = copy.deepcopy(self.wtminus1[name] - aggregation_weights)
            ps_grad_comp = copy.deepcopy(ps_grad + self.lambd * (param.data - self.wtminus1[name])
                                         * compensation * compensation).to(self.device)
            if name not in self.gminus1:
                self.gminus1[name] = copy.deepcopy(ps_grad_comp)
            # self.mom_buffer[name] = copy.deepcopy(self.mom * (self.mom_buffer[name] + ps_grad_comp
            #                                                   - self.gminus1[name]) + ps_grad_comp).to(self.device)
            self.mom_buffer[name] = copy.deepcopy(self.mom * self.mom_buffer[name] + ps_grad_comp).to(self.device)
            # self.mom_buffer[name] = self.mom * self.mom_buffer[name] + ps_grad_comp
            self.gminus1[name] = copy.deepcopy(ps_grad_comp).to(self.device)
            self.wtminus1[name] = copy.deepcopy(param.data).to(self.device)
            # param.data = param.data - self.lr * (ps_grad_comp + self.mom * self.mom_buffer[name])
            param.data = copy.deepcopy(param.data - self.mom_buffer[name]).to(self.device)
            # param.data = copy.deepcopy(aggregation_weights).to(self.device)

        for aggregation_item in self.aggregation:
            aggregation_item.clear_data()
        # print(f"PS finished updating model")

    def sync_counter(self):
        self.client_counter += 1
        # print(f"PS received data from {self.client_counter} workers")
        if self.client_counter == self.world_size - 1:
            # print(f"PS starts to aggregate data")
            self.client_counter = 0
            self.model_update()
            self.broadcast_state()

    def get_final_accuract(self):
        return max(self.acc_list)


class TrainerNet(nn.Module):
    def __init__(self, gpu, rank, world_size, dataset, model, batch_size, lr, seed, exp_id):
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        self.iid = False

        logger.add(f"logs/asyncfed/{world_size}_{dataset}_{batch_size}_{lr}"
                   f"_{exp_id}_{rank}.log")
        self.rank = rank
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.model = load_model(model, class_num=100 if dataset == "cifar100" else 10).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.)
        self.loss_func = nn.CrossEntropyLoss()
        self.sync_future_all = None
        self.train_loader , self.test_loader = partition_dataset(dataset, world_size - 1, rank - 1,
                                                                 batch_size, seed, iid=self.iid)

        self.latest_model = {}
        self.prefetch_model = {}

        self.extra_work = 0.
        self.extra_work_iter = ExtraWorkLoader(self.train_loader)

    def embedding_param_server(self, _rref):
        # print(f"Worker {self.rank} add a parameter server reference")
        self.param_server_rref = _rref
        remote_method(ParameterServer.worker_weight_update, self.param_server_rref, self.rank,
                      len(self.train_loader.dataset))

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def model_train(self, data, target):
        self.model.zero_grad()
        pred = self.model(data)
        loss = self.loss_func(pred, target)
        # if i > 0 and i % 10 == 0:
        #     logger.info(f"Rank {self.rank} training interation {i} loss: {total_loss / len(self.train_loader)}")
        loss_item = loss.item()
        loss.backward()
        self.optimizer.step()
        return loss_item

    def train_locally(self, epoch, cluster_is_ready):
        # print(f"Worker {self.rank} starts to train locally")
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = adjust_learning_rate(self.lr, epoch)
        # 1. sync data and fetch the latest model from the master
        sync_time = 0.0
        fetch_time = 0.0
        if epoch == 0:
            # we want the worker to synchronize the model with PS at first
            sync_time = self.fetch_state_block()
        if cluster_is_ready:
            self.pre_fetch_state()
            sync_time = self.synchronize()
            fetch_time = self.fetch_state()
        # 2. do local training
        total_loss = 0.
        train_start = time.time()
        for i, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            loss = self.model_train(data, target)
            total_loss += loss
        # logger.info(f"Rank {self.rank} extra work: {self.extra_work}")
        # extra_iteration = int(self.extra_work * len(self.train_loader))
        # for i in range(extra_iteration):
        #     data, target = self.extra_work_iter.retrieve()
        #     data, target = data.to(self.device), target.to(self.device)
        #     loss = self.model_train(data, target)
        #     total_loss += loss

        train_end = time.time()
        logger.info(f"Rank {self.rank} training time: {train_end - train_start}")
        logger.info(f"Rank {self.rank} training loss: {total_loss / len(self.train_loader)}")
        logger.info(f"Rank {self.rank} communication time: {fetch_time + sync_time}")
        return self.rank, train_end - train_start, fetch_time + sync_time

    def pre_fetch_state(self):
        self.prefetch_model = copy.deepcopy(self.latest_model)

    def fetch_state_block(self):
        sync_time = 0.0
        for name, param in self.model.named_parameters():
            param.data = copy.deepcopy(remote_method(
                ParameterServer.distribute_state, self.param_server_rref, name
            )).to(self.device)
            sync_time += (1.6e-06 * param.data.size().numel())
        return sync_time

    def fetch_state(self):
        # print("fetch", self.prefetch_model.keys())
        fetch_time = 0.0
        for name, param in self.model.named_parameters():
            if name not in self.prefetch_model:
                continue
            param.data = copy.deepcopy(self.prefetch_model[name]).to(self.device)
            fetch_time += (1.6e-06 * param.size().numel())
        return fetch_time

    def recv_extra_work(self, extra_work):
        self.extra_work = extra_work

    def recv_state(self, k, value):
        # print(f"Rank {self.rank} receive {k}")
        self.latest_model[k] = copy.deepcopy(value).to(self.device)
        # time.sleep(1.6e-06 * value.size().numel())

    def sync_state(self):
        if self.sync_future_all is None:
            state = True
        else:
            state = self.sync_future_all.done()
        # print(f"Worker {self.rank}'s states query returns {state}")
        return state

    def sync_wait(self):
        self.sync_future_all.wait()

    def synchronize(self):
        # print(f"Worker {self.rank} starts synchronizing")
        futs = []
        sync_time = 0.0
        for name, param in self.model.named_parameters():
            # print(f"Worker {self.rank} starts to synchronize {name}")
            sync_time += (1.6e-06 * param.size().numel())
            fut = remote_method_async(ParameterServer.synchronize, self.param_server_rref, self.rank,
                                      name, copy.deepcopy(param.data).to("cpu"))
            futs.append(fut)
        self.sync_future_all = torch.futures.collect_all(futs)
        self.sync_future_all.then(lambda x: remote_method_async(
            ParameterServer.sync_counter, self.param_server_rref
        ))
        return sync_time

