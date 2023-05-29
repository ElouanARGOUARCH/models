import torch
import torchvision.datasets as datasets

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

def get_MNIST_dataset(one_hot = False):
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