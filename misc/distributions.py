import torch

class Uniform:
    def __init__(self, lower, upper):
        self.p = upper.shape[0]
        self.lower = lower
        self.upper = upper
        assert torch.sum(upper>lower) == self.p, 'upper bound should be greater or equal to lower bound'
        self.log_scale = torch.log(self.upper - self.lower)
        self.location = (self.upper + self.lower)/2

    def log_prob(self, samples):
        condition = ((samples > self.lower).sum(-1) == self.p) * ((samples < self.upper).sum(-1) == self.p)*1
        inverse_condition = torch.logical_not(condition) * 1
        true = -torch.logsumexp(self.log_scale, dim = -1) * condition
        false = torch.nan_to_num(-torch.inf*inverse_condition, nan = 0)
        return (true + false)

    def sample(self, num_samples):
        desired_size = num_samples.copy()
        desired_size.append(self.p)
        return self.lower.expand(desired_size) + torch.rand(desired_size)*torch.exp(self.log_scale.expand(desired_size))

class Mixture:
    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.number_components = len(self.distributions)
        assert weights.shape[0] == self.number_components, 'wrong number of weights'
        self.weights = weights/torch.sum(weights)

    def log_prob(self, samples):
        list_evaluated_distributions = []
        for i,distribution in enumerate(self.distributions):
            list_evaluated_distributions.append(distribution.log_prob(samples).reshape(samples.shape[:-1]).unsqueeze(1) + torch.log(self.weights[i]))
        return(torch.logsumexp(torch.cat(list_evaluated_distributions, dim =1), dim = 1))

    def sample(self, num_samples, joint = False):
        sampled_distributions = []
        for distribution in self.distributions:
            sampled_distributions.append(distribution.sample(num_samples).unsqueeze(1))
        temp = torch.cat(sampled_distributions, dim = 1)
        pick = torch.distributions.Categorical(self.weights).sample(num_samples).squeeze(-1)
        temp2 = torch.stack([temp[i,pick[i],:] for i in range(temp.shape[0])])
        if joint:
            return temp2, pick
        else:
            return temp2

class GeneralizedMultivariateNormal():
    def __init__(self, p):
        self.p = p
        self.log_r = torch.log(torch.tensor(2.)).repeat(self.p)

    def sample(self, num_samples, r = None):
        if r is None:
            r = torch.exp(self.log_r)
        Y = torch.distributions.gamma.Gamma((1 / r), (torch.tensor(1 / 1.4142) ** r)).sample(num_samples) ** (1 / r)
        u = torch.distributions.uniform.Uniform(torch.zeros(self.p), torch.ones(self.p)).sample(num_samples)
        return (u > .5) * Y - (u < .5) * Y

    def log_density(self, x, r = None):
        if r is None:
            r = torch.exp(self.log_r)
        log_2_sqrt_2 = 1.0397
        sqrt_2 = 1.4142
        return -((torch.abs(x)/sqrt_2)**r).sum(-1) - torch.lgamma(1+1/r).sum(-1) -self.p*log_2_sqrt_2

