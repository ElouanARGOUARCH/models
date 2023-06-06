import torch
from torch import nn
import math


class ConditionalTarget(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, theta):
        raise NotImplementedError

    def log_prob(self, x, theta):
        raise NotImplementedError

class DeformedCircles(ConditionalTarget):
    def __init__(self):
        super().__init__()

    def sample(self, theta, means=torch.tensor([1., 2.]), weights=torch.tensor([.5, .5]), noise=0.125):
        angle = torch.rand(theta.shape[0]) * 2 * math.pi
        cat = torch.distributions.Categorical(weights).sample([theta.shape[0]])
        x, y = means[cat] * torch.cos(angle) + torch.randn_like(angle) * noise, means[cat] * torch.sin(
            angle) + torch.randn_like(angle) * noise
        return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)*theta

    def log_prob(self, samples,theta, means=torch.tensor([1., 2.]), weights=torch.tensor([.5, .5]), noise=0.125):
        r = torch.norm(samples/theta, dim=-1).unsqueeze(-1)
        cat = torch.distributions.Categorical(weights)
        mvn = torch.distributions.MultivariateNormal(means.unsqueeze(-1),
                                                     torch.eye(1).unsqueeze(0).repeat(2, 1, 1) * noise)
        mixt = torch.distributions.MixtureSameFamily(cat, mvn)
        return mixt.log_prob(r)

class Wave(ConditionalTarget):
    def __init__(self):
        super().__init__()
        self.p = 1
        self.d = 1

    def mu(self,theta):
        return torch.sin(math.pi * theta)/(1+theta**2)+ torch.sin(math.pi * theta / 3.0)

    def sigma2(self,theta):
        return torch.square(.5 * (1.2 - 1 / (1 + 0.1 * theta ** 2))) + 0.05

    def sample(self, thetas):
        return torch.cat([torch.distributions.Normal(self.mu(theta), self.sigma2(theta)).sample().unsqueeze(-1) for theta in thetas], dim=0)

class DoubleWave(ConditionalTarget):
    def __init__(self):
        super().__init__()
        self.p = 1
        self.d = 1

    def mu(self,theta):
        return torch.sin(math.pi * theta)/(1+theta**2)+ torch.sin(math.pi * theta / 3.0)

    def sigma2(self,theta):
        return torch.square(.5 * (1.2 - 1 / (1 + 0.1 * theta ** 2))) + 0.05

    def sample(self, thetas):
        return torch.cat([torch.distributions.MixtureSameFamily(torch.distributions.Categorical(torch.tensor([.5,.5])),torch.distributions.Normal(torch.cat([self.mu(theta),-self.mu(theta)]), torch.cat([self.sigma2(theta),self.sigma2(theta)]))).sample([1]).unsqueeze(-1) for theta in thetas], dim=0)