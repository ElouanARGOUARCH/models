import torch
import math
import matplotlib.pyplot as plt

class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self):
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

class TwoCircles(Target):
    def __init__(self):
        super().__init__()
        self.means = torch.tensor([1.,2.])
        self.weights = torch.tensor([.5, .5])
        self.noise = torch.tensor([0.125])

    def sample(self,num_samples):
        angle = torch.rand(num_samples)*2*math.pi
        cat = torch.distributions.Categorical(self.weights).sample(num_samples)
        x,y = self.means[cat]*torch.cos(angle) + torch.randn_like(angle)*self.noise,self.means[cat]*torch.sin(angle) + torch.randn_like(angle)*self.noise
        return torch.cat([x.unsqueeze(-1),y.unsqueeze(-1)], dim =-1)

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