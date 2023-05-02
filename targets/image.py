import torch
import numpy
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class image_2d_distribution():
    def __init__(self, file):
        image = plt.imread(file)
        self.image = torch.tensor(rgb2gray(image))
        self.vector_density = self.image.flatten()/torch.sum(self.image)
        self.lines, self.columns = self.image.shape

    def sample(self, num_samples):
        cat = torch.distributions.Categorical(probs=self.vector_density)
        categorical_samples = cat.sample(num_samples)
        return torch.cat([((categorical_samples % self.columns + torch.rand(num_samples)) / self.columns).unsqueeze(-1),
                                    (
                                    (1 - (categorical_samples // self.columns + torch.rand(num_samples)) / self.lines)).unsqueeze(
                                        -1)], dim=-1)

    def log_prob(self,x):
        mask = torch.logical_and(torch.all(x>0, dim= -1),torch.all(x<1, dim = -1))
        selected = x[mask]

        x_ = torch.floor(selected*torch.tensor([self.image.shape[1], self.image.shape[0]]).unsqueeze(0)).long()
        output = -torch.ones(x.shape[0])*torch.inf
        output[mask] = torch.log(self.image[self.image.shape[0]-x_[:,1], x_[:,0]]).float()
        return output


