import torch
from torch import nn
from tqdm import tqdm

class LARS_DE(nn.Module):
    def __init__(self, target_samples, hidden_dims):
        super().__init__()
        self.target_samples = target_samples
        self.p = target_samples.shape[-1]

        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        network.extend([nn.LogSigmoid(), ])

        self.log_alpha = nn.Sequential(*network)
        self.proposal = torch.distributions.MultivariateNormal(torch.mean(target_samples, dim=0),
                                                               torch.cov(target_samples.T))
        self.w = torch.distributions.Dirichlet(torch.ones(target_samples.shape[0])).sample()


    def sample(self, num_samples):
        proposed_samples = self.proposal.sample(num_samples)
        acceptance_probability = torch.exp(self.log_alpha(proposed_samples)).squeeze(-1)
        mask = torch.rand(acceptance_probability.shape) < acceptance_probability
        return proposed_samples[mask]

    def estimate_log_constant(self, num_samples):
        proposed_samples = self.proposal.sample(num_samples)
        self.log_constant = torch.logsumexp(self.log_alpha(proposed_samples).squeeze(-1), dim=0) - torch.log(
            torch.tensor([proposed_samples.shape[0]]))

    def log_prob(self, x):
        return self.proposal.log_prob(x) + self.log_alpha(x).squeeze(-1) - self.log_constant

    def loss(self, x, w):
        loss = -torch.sum(w*self.log_prob(x))
        return loss

    def train(self, epochs, batch_size = None, lr = 5e-3, weight_decay = 5e-5, verbose = False):

        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3, weight_decay = weight_decay)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        dataset = torch.utils.data.TensorDataset(self.target_samples.to(device), self.w.to(device))

        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                self.estimate_log_constant([10000])
                batch_loss = self.loss(batch[0], batch[1])
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor(
                    [self.loss(batch[0], batch[1]) for i, batch in enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 6)) + ' ; device: ' + str(device))
        self.to(torch.device('cpu'))