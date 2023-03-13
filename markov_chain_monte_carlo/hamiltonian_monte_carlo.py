import torch
from tqdm import tqdm

class HamiltonianMonteCarlo(torch.nn.Module):
    def __init__(self, target_log_prob, d,proposal_distribution=None, number_chains=1):
        super().__init__()
        self.target_log_prob = target_log_prob
        self.d = d

        if proposal_distribution is None:
            self.proposal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.proposal_distribution = proposal_distribution
        self.number_chains = number_chains

    def hamiltonian_energy(self, x, p, std):
        return torch.sum(torch.square(p), dim=-1) / (2 * (std ** 2)) - self.target_log_prob(x)

    def leapfrog(self, x, p, dt, L):
        # half grad step
        p = p + dt * torch.autograd.grad(torch.sum(self.target_log_prob(x)), x)[0] / 2
        for i in range(L):
            # full momentum step
            x = x + dt * p
            if i != L - 1:
                p = p + dt * torch.autograd.grad(torch.sum(self.target_log_prob(x)), x)[0]
        p = p + dt * torch.autograd.grad(torch.sum(self.target_log_prob(x)), x)[0] / 2
        return x, p

    def metropolis_adjustement_probability(self, x, p, x_prime, p_prime, std):
        current_hamiltonian_energy = self.hamiltonian_energy(x_prime, p_prime, std)
        initial_hamiltonian_energy = self.hamiltonian_energy(x, p, std)
        acceptance_log_prob = initial_hamiltonian_energy - current_hamiltonian_energy
        return acceptance_log_prob

    def hamiltonian_monte_carlo_step(self, x, std, dt, L):
        p = std * torch.randn_like(x)
        x.requires_grad_()
        new_x, new_p = self.leapfrog(x, p, dt, L)
        #Metropolis adjustment step
        acceptance_log_prob = self.metropolis_adjustement_probability(x, p, new_x, new_p, std)
        mask = ((torch.rand(x.shape[0]) < torch.exp(acceptance_log_prob)) * 1.).unsqueeze(-1)
        x = (mask) * new_x + (1 - (mask)) * x
        return x,mask

    def sample(self, number_steps, std=1, dt=0.01, L=20):
        x = self.proposal_distribution.sample([self.number_chains])
        pbar = tqdm(range(number_steps))
        for t in pbar:
            x,mask = self.hamiltonian_monte_carlo_step(x, std, dt, L)
            pbar.set_postfix_str('acceptance = ' + str(torch.mean(mask)))
        return x

    def sample_trajectory(self, trajectory_length, std=1, dt=0.01, L=20):
        x = self.proposal_distribution.sample([1])
        pbar = tqdm(range(trajectory_length))
        trajectory = []
        for t in pbar:
            x,mask = self.hamiltonian_monte_carlo_step(x, std, dt, L)
            trajectory.append(x)
            pbar.set_postfix_str('acceptance = ' + str(torch.mean(mask)))
        return torch.cat(trajectory, dim=0)