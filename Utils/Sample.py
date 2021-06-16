import torch
import random
import numpy as np


def mnist_iid(dataset, num_users, seed):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    return iid_part(dataset, num_users, seed)


def mnist_noniid(dataset, num_users, seed):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 600, 100
    return noniid_part(dataset, num_users, num_shards, num_imgs, seed)


def emnist_iid(dataset, num_users, seed):
    """
    Sample I.I.D. client data from EMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    return iid_part(dataset, num_users, seed)


def emnist_noniid(dataset, num_users, seed):
    """
    Sample non-I.I.D client data from EMNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 1128, 100 # 112800
    return noniid_part(dataset, num_users, num_shards, num_imgs, seed)


def cifar10_iid(dataset, num_users, seed):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    return iid_part(dataset, num_users, seed)


def cifar10_noniid(dataset, num_users, seed):
    """
        Sample non-I.I.D client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return:
        """
    num_shards, num_imgs = 500, 100
    return noniid_part(dataset, num_users, num_shards, num_imgs, seed)


def cifar100_iid(dataset, num_users, seed):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    return iid_part(dataset, num_users, seed)


def cifar100_noniid(dataset, num_users, seed):
    """
        Sample non-I.I.D client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return:
        """
    num_shards, num_imgs = 500, 100
    return noniid_part(dataset, num_users, num_shards, num_imgs, seed)


def wikitext_iid(dataset, num_users):
    data_len = len(dataset)
    dict_users = []
    sizes = [1.0 / num_users for _ in range(num_users)]
    indexes = [x for x in range(0, data_len)]
    for frac in sizes:
        part_len = int(frac * data_len)
        dict_users.append(dataset[indexes[0: part_len]])
    return dict_users


def wikitext_noniid(dataset, num_users, seed):
    data_len = len(dataset)
    dict_users = []
    sizes = gen_partition(data_len, num_users, seed)
    indexes = [x for x in range(0, data_len)]
    for part_len in sizes:
        dict_users.append(dataset[indexes[0: part_len]])
    return dict_users


def gen_partition(_sum, num_users, seed):
    _seed = seed
    mean = _sum / num_users
    variance = _sum / 2

    min_v = 1
    max_v = mean + variance
    array = [min_v] * num_users

    diff = _sum - min_v * num_users
    while diff > 0:
        np.random.seed(_seed)
        a = np.random.randint(0, num_users)
        if array[a] >= max_v:
            continue
        np.random.seed(_seed)
        delta = np.random.randint(1, diff // 4 + 4)
        array[a] += delta
        diff -= delta
        array[a] += diff if diff < 0 else 0
        _seed += 2
    return np.array(array)


def iid_part(dataset, num_users, seed):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # Random choose num_items images from all images.
        np.random.seed(seed)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid_part(dataset, num_users, num_shards, num_imgs, seed):
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    # sort the label by the target value
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    idx_shard_partition = gen_partition(num_shards, num_users, seed)
    for i in range(num_users):
        np.random.seed(seed)
        rand_set = set(np.random.choice(idx_shard, int(idx_shard_partition[i]), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users
