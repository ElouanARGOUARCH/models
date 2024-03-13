import torch
from tqdm import tqdm

class Langevin(torch.nn.Module):
    def __init__(self, target_log_density, d,proposal_distribution=None, number_chains=1):
        super().__init__()
        self.target_log_density = target_log_density
        self.d = d

        if proposal_distribution is None:
            self.proposal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.proposal_distribution = proposal_distribution
        self.number_chains = number_chains

    def unadjusted_step(self,x, tau):
        u = torch.sum(self.target_log_density(x))
        grad = torch.autograd.grad(u,x)[0]
        return x + tau * grad + (2 * tau) ** (1 / 2) * torch.randn(x.shape)


    def sample_ULA(self, number_steps, tau=0.001, verbose = False, x= None):
        if x is None:
            x = self.proposal_distribution.sample([self.number_chains])
        x.requires_grad_()
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            x = self.unadjusted_step(x,tau)
        return x

    def log_Q(self, x,x_prime, tau):
        u = torch.sum(self.target_log_density(x))
        grad = torch.autograd.grad(u, x)[0]
        return torch.distributions.MultivariateNormal(x + tau * grad, 2 * tau * torch.eye(x.shape[-1])).log_prob(x_prime)

    def metropolis_adjusted_step(self,x, tau):
        u = torch.sum(self.target_log_density(x))
        grad = torch.autograd.grad(u, x)[0]
        x_prime = x + tau * grad + (2 * tau) ** (1 / 2) * torch.randn(x.shape)

        acceptance_log_prob = self.target_log_density(x_prime) - self.target_log_density(x) - self.log_Q(x,
                                                                                                              x_prime,
                                                                                                              tau) + self.log_Q(
            x_prime, x, tau)
        mask = ((torch.rand(x.shape[0]) < torch.exp(acceptance_log_prob)) * 1.).unsqueeze(-1)
        x = (mask) * x_prime + (1 - (mask)) * x
        return x,mask

    def sample_MALA(self, number_steps, tau=0.01, verbose = True, x = None):
        if x is None:
            x = self.proposal_distribution.sample([self.number_chains])
        x.requires_grad_()
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            x,mask = self.metropolis_adjusted_step(x,tau)
            if verbose:
                pbar.set_postfix_str('acceptance = ' + str(torch.mean(mask * 1.)))
        return x