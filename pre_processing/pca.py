import torch
import sklearn
from sklearn import decomposition
import matplotlib.pyplot as plt

class PCA():
    def __init__(self,data, n_components = 'mle',visual = False):
        self.transformer = decomposition.PCA(n_components)
        self.transformer.fit(data)
        if visual:
            values = torch.tensor(self.transformer.explained_variance_ratio_)
            plt.plot(range(len(values)), values)
            plt.plot(range(len(values)), torch.cumsum(values, dim = 0))
            plt.show()

    def transform(self, data):
        return torch.tensor(self.transformer.transform(data)).float()

    def inverse_transform(self, data):
        return torch.tensor(self.transformer.inverse_transform(data)).float()