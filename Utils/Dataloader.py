import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from Utils.Sample import mnist_iid, mnist_noniid, cifar10_iid, cifar10_noniid, cifar100_iid,\
    cifar100_noniid, emnist_iid, emnist_noniid, wikitext_iid, wikitext_noniid


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, seed):
        self.dataset = dataset
        self.idxs = list(idxs)
        unique, counts = np.unique(np.array(dataset.targets)[idxs], return_counts=True)
        print(dict(zip(unique, counts)))
        random.seed(seed)
        random.shuffle(self.idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def partition_dataset(dataset, world_size, rank, batch_size, seed, iid):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if dataset == "mnist":
        train_set = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        test_set = datasets.MNIST('./data', train=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        if iid:
            dict_users = mnist_iid(train_set, world_size, seed)
        else:
            dict_users = mnist_noniid(train_set, world_size, seed)
    elif dataset == "fmnist":
        train_set = datasets.FashionMNIST('./data', train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
        test_set = datasets.FashionMNIST('./data', train=False, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
        if iid:
            dict_users = mnist_iid(train_set, world_size, seed)
        else:
            dict_users = mnist_noniid(train_set, world_size, seed)
    elif dataset == "cifar10":
        train_set = datasets.CIFAR10('./data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ]))
        test_set = datasets.CIFAR10('./data', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ]))
        if iid:
            dict_users = cifar10_iid(train_set, world_size, seed)
        else:
            dict_users = cifar10_noniid(train_set, world_size, seed)
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                      ]))
        test_set = datasets.CIFAR100('./data', train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                     ]))
        if iid:
            dict_users = cifar100_iid(train_set, world_size, seed)
        else:
            dict_users = cifar100_noniid(train_set, world_size, seed)
    elif dataset == "emnist":
        train_set = datasets.EMNIST('./data', train=True, download=True, split="balanced",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        test_set = datasets.EMNIST('./data', train=False, split="balanced",
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))

        if iid:
            dict_users = emnist_iid(train_set, world_size, seed)
        else:
            dict_users = emnist_noniid(train_set, world_size, seed)
    elif dataset == "wikitext2":
        corpus = Corpus("rnn_data/wikitext-2")
        train_set, test_set = corpus.train, corpus.test

    if dataset == "wikitext2":
        if iid:
            dict_users = wikitext_iid(train_set, world_size)
        else:
            dict_users = wikitext_noniid(train_set, world_size, seed)
        train_loader = batchify(dict_users[rank], batch_size)
        eval_batch_size = 10
        val_loader = batchify(test_set, eval_batch_size)
    else:
        print(dataset, dict_users)
        train_loader = DataLoader(DatasetSplit(train_set, dict_users[rank], seed), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        print(next(iter(train_loader))[1])


    return train_loader, val_loader


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // int(bsz)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * int(bsz))
    # Evenly divide the data across the bsz batches.
    data = data.view(int(bsz), -1).t().contiguous()
    return data