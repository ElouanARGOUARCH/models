import torch
from tqdm import tqdm


class IndependentMetropolisHastings(torch.nn.Module):
    def __init__(self, target_log_prob,d,proposal_distribution=None,number_chains=1):
        super().__init__()
        self.target_log_prob = target_log_prob
        self.d = d

        if proposal_distribution is None:
            self.proposal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.proposal_distribution = proposal_distribution
        self.number_chains = number_chains

    def independent_metropolis_step(self,x, x_prime):
        log_target_density_ratio = self.target_log_prob(x_prime) - self.target_log_prob(x)
        log_proposal_density_ratio = self.proposal_distribution.log_prob(x) - self.proposal_distribution.log_prob(x_prime)
        acceptance_log_prob = log_target_density_ratio + log_proposal_density_ratio
        mask = ((torch.rand(x_prime.shape[0]) < torch.exp(acceptance_log_prob)) * 1.).unsqueeze(-1)
        x = (mask) * x_prime + (1 - (mask)) * x
        return x,mask

    def sample(self, number_steps, verbose = False):
        x = self.proposal_distribution.sample([self.number_chains])
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            proposed_x = self.proposal_distribution.sample([self.number_chains])
            x,mask = self.independent_metropolis_step(x,proposed_x)
            if verbose:
                pbar.set_postfix_str('acceptance = ' + str(torch.mean(mask * 1.)))
        return x

class IndependentMetropolisHastingsMultipleObs(torch.nn.Module):
    def __init__(self, log_posterior,observations,d,proposal_distribution=None):
        super().__init__()
        self.target_log_prob = log_posterior
        self.observations = observations
        self.number_obs = observations.shape[0]
        self.d = d

        if proposal_distribution is None:
            self.proposal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.proposal_distribution = proposal_distribution

    def independent_metropolis_step(self,x, x_prime):
        log_target_density_ratio = self.target_log_prob(x_prime, self.observations) - self.target_log_prob(x, self.observations)
        log_proposal_density_ratio = self.proposal_distribution.log_prob(x) - self.proposal_distribution.log_prob(x_prime)
        acceptance_log_prob = log_target_density_ratio + log_proposal_density_ratio
        mask = ((torch.rand(x_prime.shape[0]) < torch.exp(acceptance_log_prob)) * 1.).unsqueeze(-1)
        x = (mask) * x_prime + (1 - (mask)) * x
        return x,mask

    def sample(self, number_steps, verbose = False):
        x = self.proposal_distribution.sample([self.number_obs])
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            proposed_x = self.proposal_distribution.sample([self.number_obs])
            x,mask = self.independent_metropolis_step(x,proposed_x)
            if verbose:
                pbar.set_postfix_str('acceptance = ' + str(torch.mean(mask * 1.)))
        return x

class RandomWalkMetropolisHastings(torch.nn.Module):
    def __init__(self, target_log_prob,d,sigma2=1.,number_chains=1, initial_distribution = None):
        super().__init__()
        self.target_log_prob = target_log_prob
        self.d = d
        self.sigma2 = sigma2

        if initial_distribution is None:
            self.initial_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.initial_distribution = initial_distribution
        self.number_chains = number_chains

    def independent_metropolis_step(self,x, x_prime):
        log_target_density_ratio = self.target_log_prob(x_prime) - self.target_log_prob(x)
        acceptance_log_prob = log_target_density_ratio
        mask = ((torch.rand(x_prime.shape[0]) < torch.exp(acceptance_log_prob)) * 1.).unsqueeze(-1)
        x = (mask) * x_prime + (1 - (mask)) * x
        return x,mask

    def sample(self, number_steps, verbose = False):
        x = self.initial_distribution.sample([self.number_chains])
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            proposed_x = x + torch.randn_like(x)*self.sigma
            x,mask = self.independent_metropolis_step(x,proposed_x)
            if verbose:
                pbar.set_postfix_str('acceptance = ' + str(torch.mean(mask * 1.)))
        return x
