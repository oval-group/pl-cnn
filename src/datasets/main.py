from datasets.__local__ import implemented_datasets
from datasets.cifar10 import CIFAR_10_DataLoader
from datasets.cifar100 import CIFAR_100_DataLoader
from datasets.imagenet import ImageNet_DataLoader
from datasets.mnist import MNIST_DataLoader


def load_dataset(nnet, dataset_name):

    assert dataset_name in implemented_datasets

    if dataset_name == "mnist":
        data_loader = MNIST_DataLoader

    if dataset_name == "cifar10":
        data_loader = CIFAR_10_DataLoader

    if dataset_name == "cifar100":
        data_loader = CIFAR_100_DataLoader

    if dataset_name == "imagenet":
        data_loader = ImageNet_DataLoader

    # load data with data loader
    nnet.load_data(data_loader=data_loader)

    # check all parameters have been attributed
    nnet.data.check_all()
