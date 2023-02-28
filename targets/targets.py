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

    def sample(self,num_samples, means = torch.tensor([1.,2.]),weights = torch.tensor([.5,.5]), noise = 0.125):
        angle = torch.rand(num_samples)*2*math.pi
        cat = torch.distributions.Categorical(weights).sample(num_samples)
        x,y = means[cat]*torch.cos(angle) + torch.randn_like(angle)*noise,means[cat]*torch.sin(angle) + torch.randn_like(angle)*noise
        return torch.cat([x.unsqueeze(-1),y.unsqueeze(-1)], dim =-1)

    def log_prob(self,samples, means = torch.tensor([1.,2.]),weights = torch.tensor([.5,.5]), noise = 0.125):
        r = torch.norm(samples, dim=-1).unsqueeze(-1)
        cat = torch.distributions.Categorical(weights)
        mvn = torch.distributions.MultivariateNormal(means.unsqueeze(-1), torch.eye(1).unsqueeze(0).repeat(2,1,1)*noise)
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
        return self.mix_target.sample(num_samples)

    def log_prob(self,x):
        return self.mix_target.log_prob(x)