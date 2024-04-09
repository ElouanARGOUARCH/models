import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


def shuffle(tensor, randperm=None):
    if randperm is None:
        randperm = torch.randperm(tensor.shape[0])
    return tensor[randperm], randperm

def get_MNIST_dataset(one_hot = False,repository = 'C:\\Users\\Elouan\\PycharmProjects\\models\\targets\\data', visual = True):
    mnist_trainset = datasets.MNIST(root=repository, train=True,
                                    download=True, transform=None)
    mnist_testset = datasets.MNIST(root=repository, train=False,
                                   download=True, transform=None)
    train_labels = mnist_trainset.targets
    test_labels = mnist_testset.targets
    temp_train = mnist_trainset.data.flatten(start_dim=1).float()
    train_samples = (temp_train + torch.rand_like(temp_train))/256
    temp_test = mnist_testset.data.flatten(start_dim=1).float()
    test_samples = (temp_test + torch.rand_like(temp_test))/256
    if visual:
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            plt.imshow(train_samples[train_labels==i,:][0].reshape(28,28))
        # show the figure
        plt.show()
    if one_hot:
        return torch.cat([train_samples, test_samples], dim = 0), torch.nn.functional.one_hot(torch.cat([train_labels,test_labels], dim = 0))
    else:
        return torch.cat([train_samples, test_samples], dim = 0), torch.cat([train_labels,test_labels], dim = 0)

_ = get_MNIST_dataset()

def get_FashionMNIST_dataset(one_hot = False,repository = 'C:\\Users\\Elouan\\PycharmProjects\\models\\targets\\data', visual = True):
    fmnist_trainset = datasets.FashionMNIST(root=repository, train=True,
                                            download=True, transform=None)
    fmnist_testset = datasets.FashionMNIST(root=repository, train=False,
                                           download=True, transform=None)
    train_labels = fmnist_trainset.targets
    test_labels = fmnist_testset.targets
    temp_train = fmnist_trainset.data.flatten(start_dim=1).float()
    train_samples = (temp_train + torch.rand_like(temp_train))/256
    temp_test = fmnist_testset.data.flatten(start_dim=1).float()
    test_samples = (temp_test + torch.rand_like(temp_test))/256
    if visual:
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            plt.imshow(train_samples[train_labels==i,:][0].reshape(28,28))
        # show the figure
        plt.show()
    if one_hot:
        return torch.cat([train_samples, test_samples], dim = 0), torch.nn.functional.one_hot(torch.cat([train_labels,test_labels], dim = 0))
    else:
        return torch.cat([train_samples, test_samples], dim = 0), torch.cat([train_labels,test_labels], dim = 0)



from sklearn.datasets import load_digits
def get_DIGITS_dataset(one_hot = False):
    samples, labels = load_digits(return_X_y=True)
    samples = torch.tensor(samples).float()
    labels = torch.tensor(labels).to(torch.int64)
    if one_hot:
        return (samples + torch.rand_like(samples))/17, torch.nn.functional.one_hot(torch.tensor(labels))
    else:
        return (samples + torch.rand_like(samples))/17, torch.tensor(labels)

from ucimlrepo import fetch_ucirepo
def get_WineQuality_dataset(one_hot = False):
    wine_quality = fetch_ucirepo(id=186)
    samples = wine_quality.data.features
    labels = wine_quality.data.targets
    if one_hot:
        return torch.tensor(samples), torch.nn.functional.one_hot(torch.tensor(labels))
    else:
        return torch.tensor(samples),torch.tensor(labels)
'''
from keras.datasets import cifar10, cifar100
import matplotlib.pyplot as plt
def get_CIFAR10_dataset(one_hot = False, visual = False):
    (trainX, trainy), (testX, testy) = cifar10.load_data()
    samples = torch.cat([torch.tensor(trainX),torch.tensor(testX)], dim = 0)
    labels = torch.cat([torch.tensor(trainy),torch.tensor(testy)], dim = 0).squeeze(-1).long()
    samples = samples.reshape(samples.shape[0], 32*32*3)
    if one_hot:
        labels = torch.nn.functional.one_hot(labels,num_classes =10)
    if visual:
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            plt.imshow(trainX[i])
        # show the figure
        plt.show()
    return samples, labels
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def get_CIFAR100_dataset(one_hot = False, visual = False):
    (trainX, trainy), (testX, testy) = cifar100.load_data()
    samples = torch.cat([torch.tensor(trainX),torch.tensor(testX)], dim = 0)
    labels = torch.cat([torch.tensor(trainy),torch.tensor(testy)], dim = 0).squeeze(-1).long()
    samples = samples.reshape(samples.shape[0], 32*32*3)
    if one_hot:
        labels = torch.nn.functional.one_hot(labels,num_classes =100)
    if visual:
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            plt.imshow(trainX[i])
        # show the figure
        plt.show()
    return samples, labels'''