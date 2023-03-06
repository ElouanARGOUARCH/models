import torch
from tqdm import tqdm

from density_estimation.discretely_indexed_flows import SoftmaxWeight, LocationScaleFlow

class DIFSampler(torch.nn.Module):
    def __init__(self, target_log_prob, p, K, hidden_dims = []):
        super().__init__()
        self.target_log_prob = target_log_prob
        self.p = p
        self.K = K

        self.w = SoftmaxWeight(self.K, self.p, hidden_dims)

        self.T = LocationScaleFlow(self.K, self.p)

    def reference_log_prob(self, z):
        return torch.distributions.MultivariateNormal(torch.zeros(self.p).to(z.device), torch.eye(self.p).to(z.device)).log_prob(z)

    def compute_log_v(self, x):
        z = self.T.forward(x)
        log_v = self.reference_log_prob(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim=-1, keepdim=True)

    def DKL_observed(self, z):
        x = self.T.backward(z)
        return torch.mean(torch.sum(torch.exp(self.w.log_prob(z))*(self.log_prob(x) - self.target_log_prob(x)),dim = -1))

    def DKL_latent(self,z):
        return torch.mean(self.reference_log_prob(z) - self.latent_log_prob(z))

    def sample(self, num_samples):
        z = torch.randn(num_samples+[self.p])
        x = self.T.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.w.log_prob(z))).sample()
        return x[range(x.shape[0]), pick, :]

    def latent_log_prob(self, z):
        x = self.T.backward(z)
        return torch.logsumexp(torch.diagonal(self.compute_log_v(x), 0, -2, -1) + self.target_log_prob(x) - self.T.log_det_J(x), dim=-1)

    def log_prob(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.reference_log_prob(z) + self.T.log_det_J(x), dim=-1)

    def train(self, epochs,num_samples,lr = 5e-3, weight_decay = 5e-5, verbose = False):
        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr, weight_decay)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            z = torch.randn([num_samples,self.p]).to(device)
            self.optimizer.zero_grad()
            loss = self.DKL_observed(z)
            loss.backward()
            self.optimizer.step()
            if verbose:
                with torch.no_grad():
                    DKL_latent = self.DKL_latent(z)
                self.loss_values.append(loss)
                pbar.set_postfix_str('DKL observed = ' + str(round(loss.item(), 6)) + ' DKL Latent = ' + str(
                    round(DKL_latent.item(), 6)) + ' ; device: ' + str(device))
        self.to(torch.device('cpu'))