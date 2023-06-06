import torch
import sklearn
from sklearn import datasets
import math
import matplotlib.pyplot as plt

class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        raise NotImplementedError

    def log_prob(self):
        raise NotImplementedError

    def visual(self, num_samples = 5000):
        samples = self.sample([num_samples])
        if samples.shape[-1] == 1:
            plt.figure(figsize=(8, 8))
            plt.hist(samples[:, 0].numpy(), bins=150, color='red',density = True, alpha=0.6)
            plt.show()
        if samples.shape[-1] >= 2:
            plt.figure(figsize=(8, 8))
            plt.scatter(samples[:,-2], samples[:,-1],color='red',alpha=0.6)
            plt.show()

class SCurve(Target):
    def __init__(self):
        super().__init__()

    def sample(self, num_samples):
        X, t = datasets.make_s_curve(num_samples[0], noise=0.05)
        X = torch.tensor(sklearn.preprocessing.StandardScaler().fit_transform(X)).float()
        return torch.cat([X[:,0].unsqueeze(-1), X[:,-1].unsqueeze(-1)], dim = -1)

class TwoCircles(Target):
    def __init__(self):
        super().__init__()
        self.means = torch.tensor([1.,2.])
        self.weights = torch.tensor([.5, .5])
        self.noise = torch.tensor([0.125])

    def sample(self,num_samples, joint = False):
        angle = torch.rand(num_samples)*2*math.pi
        cat = torch.distributions.Categorical(self.weights).sample(num_samples)
        x,y = self.means[cat]*torch.cos(angle) + torch.randn_like(angle)*self.noise,self.means[cat]*torch.sin(angle) + torch.randn_like(angle)*self.noise
        if not joint:
            return torch.cat([x.unsqueeze(-1),y.unsqueeze(-1)], dim =-1)
        else:
            return torch.cat([x.unsqueeze(-1),y.unsqueeze(-1)], dim =-1), cat

    def log_prob(self,x):
        r = torch.norm(x, dim=-1).unsqueeze(-1)
        cat = torch.distributions.Categorical(self.weights.to(x.device))
        mvn = torch.distributions.MultivariateNormal(self.means.to(x.device).unsqueeze(-1), torch.eye(1).to(x.device).unsqueeze(0).repeat(2,1,1)*self.noise.to(x.device))
        mixt = torch.distributions.MixtureSameFamily(cat, mvn)
        return mixt.log_prob(r)

class Orbits(Target):
    def __init__(self):
        super().__init__()
        number_planets = 7
        covs_target = 0.04 * torch.eye(2).unsqueeze(0).repeat(number_planets, 1, 1)
        means_target = 2.5 * torch.view_as_real(
            torch.pow(torch.exp(torch.tensor([2j * 3.1415 / number_planets])), torch.arange(0, number_planets)))
        weights_target = torch.ones(number_planets)
        weights_target = weights_target

        mvn_target = torch.distributions.MultivariateNormal(means_target, covs_target)
        cat = torch.distributions.Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
        self.mix_target = torch.distributions.MixtureSameFamily(cat, mvn_target)

    def sample(self, num_samples):
        number_planets = 7
        covs_target = 0.04 * torch.eye(2).unsqueeze(0).repeat(number_planets, 1, 1)
        means_target = 2.5 * torch.view_as_real(
            torch.pow(torch.exp(torch.tensor([2j * 3.1415 / number_planets])), torch.arange(0, number_planets)))
        weights_target = torch.ones(number_planets)

        mvn_target = torch.distributions.MultivariateNormal(means_target, covs_target)
        cat = torch.distributions.Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
        return torch.distributions.MixtureSameFamily(cat, mvn_target).sample(num_samples)

    def log_prob(self,x):
        number_planets = 7
        covs_target = 0.04 * torch.eye(2).unsqueeze(0).repeat(number_planets, 1, 1)
        means_target = 2.5 * torch.view_as_real(
            torch.pow(torch.exp(torch.tensor([2j * 3.1415 / number_planets])), torch.arange(0, number_planets)))
        weights_target = torch.ones(number_planets).to(x.device)

        mvn_target = torch.distributions.MultivariateNormal(means_target.to(x.device), covs_target.to(x.device))
        cat = torch.distributions.Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
        return torch.distributions.MixtureSameFamily(cat, mvn_target).log_prob(x)

class Banana(Target):
    def __init__(self):
        super().__init__()
        var = 2
        dim = 50
        self.even = torch.arange(0, dim, 2)
        self.odd = torch.arange(1, dim, 2)
        self.mvn = torch.distributions.MultivariateNormal(torch.zeros(dim), var * torch.eye(dim))

    def transform(self, x):
        z = x.clone()
        z[...,self.odd] += z[...,self.even]**2
        return z

    def sample(self, num_samples):
        return self.transform(self.mvn.sample([num_samples]))

    def log_prob(self, samples):
        return self.mvn.log_prob(self.inv_transform(samples))

class Funnel(Target):
    def __init__(self):
        super().__init__()
        self.a = torch.tensor(1.)
        self.b = torch.tensor(0.5)
        self.dim = 20

        self.distrib_x1 = torch.distributions.Normal(torch.zeros(1), torch.tensor(self.a))

    def sample(self, num_samples):
        x1 = self.distrib_x1.sample([num_samples])

        rem = torch.randn((num_samples,) + (self.dim - 1,)) * (self.b * x1).exp()

        return torch.cat([x1, rem], -1)
    def log_prob(self, x):
        log_probx1 = self.distrib_x1.log_prob(x[..., 0].unsqueeze(1))
        logprob_rem = (- x[..., 1:] ** 2 * (-2 * self.b * x[..., 0].unsqueeze(-1)).exp() - 2 * self.b * x[:, 0].unsqueeze(-1) - torch.tensor(2 * math.pi).log()) / 2
        logprob_rem = logprob_rem.sum(-1)
        return (log_probx1 + logprob_rem.unsqueeze(-1)).flatten()

class Well(Target):
    def __init__(self,d = 2):
        super().__init__()
        self.d = d
        self.number_spikes = 30
        self.weights = torch.distributions.Dirichlet(torch.ones(self.number_spikes)).sample()
        self.means = torch.rand(self.number_spikes, self.d) - torch.ones(self.number_spikes, self.d)*0.5
        self.vars = torch.rand(self.number_spikes, self.d,1)*torch.eye(self.d).unsqueeze(0).repeat(self.number_spikes,1,1)
        self.total_mean = torch.cat([torch.zeros(1,self.d), self.means], dim = 0)
        self.total_var = torch.cat([torch.eye(self.d).unsqueeze(0), self.vars], dim =0)
        self.mvn = torch.distributions.MultivariateNormal(self.total_mean, self.total_var)
        self.total_weights = torch.cat([torch.tensor([self.number_spikes]), self.weights], dim = 0)
        print(self.total_weights)
        self.cat = torch.distributions.Categorical(self.total_weights/torch.tensor(self.total_weights))
        self.mixt = torch.distributions.MixtureSameFamily(self.cat, self.mvn)

    def sample(self,num_samples):
        return self.mixt.sample(num_samples)

    def log_prob(self, x):
        return self.mixt.log_prob(x)

from utils import *
target = Well()
plot_2d_function(lambda x:torch.exp(target.log_prob(x)), range =((-4,4),(-4,4)), bins = (300,300))
plt.show()