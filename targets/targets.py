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
    def __init__(self, number_planets=7, means_target=None, covs_target=None, weights_target=None):
        super().__init__()
        self.number_planets = number_planets
        if means_target is None:
            self.means_target = 2.5 * torch.view_as_real(
                torch.pow(torch.exp(torch.tensor([2j * 3.1415 / number_planets])), torch.arange(0, number_planets)))
        else:
            assert means_target.shape != [self.number_planets, 2], "wrong size of means"
            self.means_target = means_target

        if covs_target is None:
            self.covs_target = 0.04 * torch.eye(2).unsqueeze(0).repeat(number_planets, 1, 1)
        else:
            assert covs_target.shape != [self.number_planets, 2, 2], 'wrong size of covs'
            self.covs_target = covs_target

        if weights_target is None:
            self.weights_target = torch.ones(self.number_planets)
        else:
            assert weights_target.shape != [self.number_planets], 'wrong size of weights'
            self.weights_target = weights_target

    def sample(self, num_samples, joint=False):
        mvn_target = torch.distributions.MultivariateNormal(self.means_target, self.covs_target)
        all_x = mvn_target.sample(num_samples)
        cat = torch.distributions.Categorical(self.weights_target.softmax(dim=0))
        pick = cat.sample(num_samples)
        if joint:
            return all_x[range(all_x.shape[0]), pick, :], pick
        else:
            return all_x[range(all_x.shape[0]), pick, :]

    def log_prob(self, x):
        mvn_target = torch.distributions.MultivariateNormal(self.means_target.to(x.device),
                                                            self.covs_target.to(x.device))
        cat = torch.distributions.Categorical(self.weights_target.softmax(dim=0).to(x.device))
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