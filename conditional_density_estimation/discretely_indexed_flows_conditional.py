import torch
from tqdm import tqdm
from density_estimation import SoftmaxWeight
from density_estimation.gaussian_mixture_model import DiagGaussianMixtEM

class ConditionalLocationScale(torch.nn.Module):
    def __init__(self, K, p, d, hidden_dimensions):
        super().__init__()
        self.K = K
        self.p = p
        self.d = d

        self.network_dimensions = [self.d] + hidden_dimensions + [2*self.K*self.p]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1),torch.nn.Tanh(),])
        network.pop()
        self.f = torch.nn.Sequential(*network)

    def backward(self, z, theta):
        assert z.shape[:-1]==theta.shape[:-1], 'number of z samples does not match the number of theta samples'
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.p
        out = torch.reshape(self.f(theta), new_desired_size)
        m, log_s = out[...,:self.p], out[...,self.p:]
        return Z * torch.exp(log_s).expand_as(Z) + m.expand_as(Z)

    def forward(self, x, theta):
        assert x.shape[:-1]==theta.shape[:-1], 'number of x samples does not match the number of theta samples'
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.p
        out = torch.reshape(self.f(theta), new_desired_size)
        m, log_s = out[...,:self.p], out[...,self.p:]
        return (X-m.expand_as(X))/torch.exp(log_s).expand_as(X)

    def log_det_J(self,x, theta):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.p
        log_s = torch.reshape(self.f(theta), new_desired_size)[..., self.p:]
        return -log_s.sum(-1)

class ConditionalDIF(torch.nn.Module):
    def __init__(self, D_x, D_theta, K, hidden_dimensions):
        super().__init__()
        self.D_x = D_x
        self.D_theta = D_theta
        assert D_theta.shape[0] == D_x.shape[0], 'number of X samples does not match the number of theta samples'
        self.p = D_x.shape[-1]
        self.d = D_theta.shape[-1]
        self.K = K

        self.w = torch.distributions.Dirichlet(torch.ones(self.D_x.shape[0])).sample()

        self.reference_mean = torch.mean(D_x, dim = 0)
        _ = torch.cov(D_x.T)
        self.reference_cov = (_.T + _)/2
        self.reference = torch.distributions.MultivariateNormal(self.reference_mean, self.reference_cov)

        self.W = SoftmaxWeight(self.K, self.p+self.d, hidden_dimensions)

        self.T = ConditionalLocationScale(self.K, self.p, self.d, hidden_dimensions)

    def initialize_with_EM(self, epochs, verbose = False):
        em = DiagGaussianMixtEM(self.D_x,self.K)
        em.train(epochs, verbose)
        self.T.f[-1].weight = torch.nn.Parameter(torch.zeros(self.T.network_dimensions[-1],self.T.network_dimensions[-2]))
        self.T.f[-1].bias = torch.nn.Parameter(torch.cat([em.m,em.log_s], dim = -1).flatten())
        self.W.f[-1].weight = torch.nn.Parameter(torch.zeros(self.W.network_dimensions[-1],self.W.network_dimensions[-2]))
        self.W.f[-1].bias = torch.nn.Parameter(em.log_pi)
        self.reference_mean = torch.zeros(self.p)
        self.reference_cov = torch.eye(self.p)

    def reference_log_prob(self,z):
        return torch.distributions.MultivariateNormal(self.reference_mean.to(z.device), self.reference_cov.to(z.device)).log_prob(z)

    def compute_log_v(self,x, theta):
        assert x.shape[:-1] == theta.shape[:-1], 'wrong shapes'
        theta_unsqueezed = theta.unsqueeze(-2).repeat(1, self.K, 1)
        z = self.T.forward(x, theta)
        log_v = self.reference.log_prob(z) + torch.diagonal(self.W.log_prob(torch.cat([z, theta_unsqueezed], dim = -1)), 0, -2, -1) + self.T.log_det_J(x, theta)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x, theta):
        assert x.shape[:-1] == theta.shape[:-1], 'wrong shapes'
        z = self.T.forward(x, theta)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x, theta))).sample()
        return z[range(z.shape[0]), pick, :]

    def log_prob(self, x, theta):
        assert x.shape[:-1] == theta.shape[:-1], 'wrong shapes'
        desired_size = list(theta.shape)
        desired_size.insert(-1, self.K)
        theta_unsqueezed = theta.unsqueeze(-2).expand(desired_size)
        z = self.T.forward(x, theta)
        return torch.logsumexp(self.reference_log_prob(z) + torch.diagonal(self.W.log_prob(torch.cat([z, theta_unsqueezed], dim = -1)), 0, -2, -1)+ self.T.log_det_J(x, theta),dim=-1)

    def sample(self, theta):
        z = torch.distributions.MultivariateNormal(self.reference_mean, self.reference_cov).sample(theta.shape[:-1])
        x = self.T.backward(z, theta)
        pick = torch.distributions.Categorical(torch.exp(self.W.log_prob(torch.cat([z, theta], dim = -1)))).sample()
        return x[range(x.shape[0]), pick, :]

    def loss(self, x, theta, w):
        return -torch.sum(w*self.log_prob(x,theta))

    def train(self, epochs, batch_size = None,lr = 5e-3, weight_decay = 5e-5,verbose = False):
        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=lr, weight_decay= weight_decay)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if batch_size is None:
            batch_size = self.D_x.shape[0]
        dataset = torch.utils.data.TensorDataset(self.D_x.to(device), self.D_theta.to(device), self.w.to(device))

        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                batch_loss = self.loss(batch[0], batch[1], batch[2])
                batch_loss.backward()
                self.optimizer.step()
            if verbose:
                with torch.no_grad():
                    iteration_loss = torch.tensor(
                        [self.loss(batch[0], batch[1], batch[2]) for i, batch in
                         enumerate(dataloader)]).sum().item()
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(device))
        self.to(torch.device('cpu'))