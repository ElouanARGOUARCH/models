import torch
from tqdm import tqdm
from density_estimation.discretely_indexed_flows import SoftmaxWeight, LocationScaleFlow

class TMC(torch.nn.Module):
    def __init__(self, target_log_prob, p, K,hidden_dims = []):
        super().__init__()
        self.target_log_prob = target_log_prob
        self.p = p
        self.K = K

        self.v = SoftmaxWeight(self.K, self.p, hidden_dims)

        self.T = LocationScaleFlow(self.K, self.p)

        self.loss_values = []
        self.para_list = list(self.parameters())

    def reference_log_prob(self, z):
        return torch.distributions.MultivariateNormal(torch.zeros(self.p).to(z.device), torch.eye(self.p).to(z.device)).log_prob(z)

    def compute_log_w(self, z):
        x = self.T.backward(z)
        log_v = self.target_log_prob(x) + torch.diagonal(self.v.log_prob(x), 0, -2, -1) - self.T.log_det_J(z)
        return log_v - torch.logsumexp(log_v, dim=-1, keepdim=True)

    def DKL_observed(self, z):
        x = self.T.backward(z)
        return torch.mean(torch.sum(torch.exp(self.compute_log_w(z))*(self.log_prob(x) - self.target_log_prob(x)),dim = -1))

    def DKL_latent(self,z):
        return torch.mean(self.reference_log_prob(z) - self.latent_log_prob(z))

    def sample(self, num_samples):
        z = torch.randn(num_samples+[self.p])
        x = self.T.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_w(z))).sample()
        return x[range(x.shape[0]), pick, :]

    def log_prob(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(torch.diagonal(self.compute_log_w(z), 0, -2, -1) + self.reference_log_prob(z) + self.T.log_det_J(z), dim=-1)

    def latent_log_prob(self, z):
        x = self.T.backward(z)
        return torch.logsumexp(torch.diagonal(self.v.log_prob(x), 0, -2, -1) + self.target_log_prob(x) - self.T.log_det_J(z), dim=-1)

    def train(self, epochs,num_samples,lr = 5e-3, weight_decay = 5e-5, verbose = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.para_list, lr, weight_decay)
        self.to(device)
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            z = torch.randn([num_samples, self.p]).to(device)
            self.optimizer.zero_grad()
            loss = self.DKL_latent(z)
            loss.backward()
            self.optimizer.step()
            if verbose:
                with torch.no_grad():
                    DKL_observed = self.DKL_observed(z)
                pbar.set_postfix_str('DKL latent = ' + str(round(loss.item(), 6)) + ' DKL observed = ' + str(
                    round(DKL_observed.item(), 6)) + ' ; device: ' + str(device))
        self.to(torch.device('cpu'))