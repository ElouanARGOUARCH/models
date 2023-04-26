import torch
import sklearn
import matplotlib.pyplot as plt

class PCA():
    def __init__(self,data, n_components = 'mle',visual = False):
        self.transformer = sklearn.decomposition.PCA(n_components)
        self.transformer.fit(data)
        if visual:
            values = self.tool.explained_variance_ratio_
            plt.plot(range(len(values)), values)
            plt.plot(range(len(values)), torch.cumsum(values))
            plt.show()

    def transform(self, data):
        return torch.tensor(self.transformer.transform(data)).float()

    def inverse_transform(self, data):
        return torch.tensor(self.transformer.inverse_transform(data)).float()