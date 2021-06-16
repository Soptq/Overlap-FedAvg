from torchvision import datasets

datasets.MNIST('./data', train=True, download=True)
datasets.FashionMNIST('./data', train=True, download=True)
datasets.CIFAR10('./data', train=True, download=True)
datasets.CIFAR100('./data', train=True, download=True)
datasets.EMNIST('./data', train=True, download=True, split="balanced")