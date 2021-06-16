import argparse
import os, random, time

import numpy as np

import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from Utils.Helper import remote_method, remote_method_async
from Characters.Characters import ParameterServer, TrainerNet


def get_parameter_server(gpu, world_size, dataset, batch_size, lr,
                         model, max_epoch, client_epoch, seed, exp_id):
    param_server = ParameterServer(gpu=gpu, world_size=world_size, dataset=dataset, batch_size=batch_size,
                                   lr=lr, model=model, max_epoch=max_epoch, client_epoch=client_epoch,
                                   seed=seed, exp_id=exp_id)
    return param_server


def get_worker(gpu, rank, world_size, dataset, model, batch_size, lr, client_epoch, seed, exp_id):
    train_server = TrainerNet(gpu=gpu, rank=rank, world_size=world_size, dataset=dataset, model=model,
                              batch_size=batch_size, lr=lr, client_epoch=client_epoch, seed=seed, exp_id=exp_id)
    return train_server


def run_driver(rank, world_size, gpu_list, dataset, batch_size, lr,
               max_epoch, client_epoch, model, seed):
    exp_id = str(int(time.time()))
    print(f"Driver initializing RPC, rank {rank}, world size {world_size}")
    rpc.init_rpc(name="driver", rank=rank, world_size=world_size)
    print("Initialized driver")
    param_server_rref = rpc.remote("parameter_server", get_parameter_server,
                                   args=(gpu_list[0], world_size - 1, dataset, batch_size,
                                         lr, model, max_epoch, client_epoch, seed, exp_id))
    for _rank in range(1, world_size - 1):
        print(f"Driver registering worker node {_rank}")
        worker_server_rref = rpc.remote(f"trainer_{_rank}", get_worker,
                                        args=(gpu_list[_rank], _rank, world_size - 1, dataset,
                                              model, batch_size, lr, client_epoch, seed, exp_id))
        print(f"Driver binding worker {_rank} with param server")
        remote_method(ParameterServer.embedding_workers, param_server_rref, worker_server_rref)
        remote_method(TrainerNet.embedding_param_server, worker_server_rref, param_server_rref)

    fut = remote_method_async(ParameterServer.instruct_training, param_server_rref)
    fut.wait()
    rpc.shutdown()
    print("RPC shutdown on Driver")


def run_parameter_server(rank, world_size):
    print(f"PS master initializing RPC, rank {rank}, world size {world_size}")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("Parameter server done initializing RPC")
    rpc.shutdown()
    print("RPC shutdown on parameter server")


def run_worker(rank, world_size):
    print(f"Worker initializing RPC, rank {rank}, world size {world_size}")
    rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=world_size)
    print(f"Worker {rank} done initializing RPC")
    rpc.shutdown()
    print(f"RPC shutdown on Worker {rank}.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Averaging Experiments")
    parser.add_argument("--world_size", type=int, default=5,
                        help="""Total number of participating processes.
                         Should be the sum of master node and all training nodes.""")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset used to participate training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for individual client")
    parser.add_argument("--max_epoch", type=int, default=10, help="Specify how many epochs will the cluster trains for")
    parser.add_argument("--client_epoch", type=int, default=5, help="Specify how many epochs will the client"
                                                                    "train for at each cluster epoch")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate of the cluster")
    parser.add_argument("--model", type=str, default="mnistnet", help="Specify the model to train.")
    parser.add_argument("--gpus", type=str, default="0,0,0,0,0", help="""Input GPU for training""")
    parser.add_argument("--master_addr", type=str, default="localhost",
                        help="""Address of master, will default to localhost if not provided.
                        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument("--master_port", type=str, default="29500",
                        help="""Port that master is listening on, will default to 29500 if not 
                        provided. Master must be able to accept network traffic on the host and port.""")
    parser.add_argument("--seed", type=int, default=2020, help="The seed of random function")

    args = parser.parse_args()

    assert args.dataset in ["mnist", "fmnist", "cifar10", "cifar100", "emnist", "wikitext2"], \
        f"Dataset can only be one of mnist, fmnist, cifar10, cifar100, emnist, wikitext2"
    assert args.model in [""mnistnet", "resnet", "vgg", "mlp", "cnncifar", "transformer"], \
        f"Model can only be one of `mnistnet`, `resnet`, `vgg`, `mlp`, `cnncifar`, `transformer`"

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    processes = []
    world_size = args.world_size + 1
    for rank in range(world_size):
        if rank == 0:
            p = mp.Process(target=run_parameter_server, args=(rank, world_size))
            p.start()
            processes.append(p)
        elif rank == world_size - 1:
            pass
            p = mp.Process(target=run_driver, args=(rank, world_size, args.gpus.split(","), args.dataset,
                                                    args.batch_size, args.lr, args.max_epoch, args.client_epoch,
                                                    args.model, args.seed))
            p.start()
            processes.append(p)
        else:
            p = mp.Process(target=run_worker, args=(rank, world_size))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
