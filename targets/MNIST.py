import torch
import torchvision.datasets as datasets

def get_MNIST_dataset(one_hot = False,repository = 'C:\Users\Elouan\PycharmProjects\models\targets\data'):
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
    if one_hot:
        return torch.cat([train_samples, test_samples], dim = 0), torch.nn.functional.one_hot(torch.cat([train_labels,test_labels], dim = 0))
    else:
        return torch.cat([train_samples, test_samples], dim = 0), torch.cat([train_labels,test_labels], dim = 0)



def get_FashionMNIST_dataset(one_hot = False,repository = 'C:\Users\Elouan\PycharmProjects\models\targets\data'):
    fmnist_trainset = datasets.FashionMNIST(root=repository, train=True,
                                            download=True, transform=None)
    fmnist_testset = datasets.FashionMNIST(root=repository, train=False,
                                           download=True, transform=None)
    train_labels = mnist_trainset.targets
    test_labels = mnist_testset.targets
    temp_train = fmnist_trainset.data.flatten(start_dim=1).float()
    train_samples = (temp_train + torch.rand_like(temp_train))/256
    temp_test = fmnist_testset.data.flatten(start_dim=1).float()
    test_samples = (temp_test + torch.rand_like(temp_test))/256
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
