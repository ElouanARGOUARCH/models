import torch
from tqdm import tqdm
from density_estimation.gaussian_mixture_model import DiagGaussianMixtEM

class SoftmaxWeight(torch.nn.Module):
    def __init__(self, K, p, hidden_dims =[]):
        super().__init__()
        self.K = K
        self.p = p
        self.network_dimensions = [self.p] + hidden_dims + [self.K]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1),torch.nn.Tanh(),])
        network.pop()
        self.f = torch.nn.Sequential(*network)

    def log_prob(self, z):
        unormalized_log_w = self.f.forward(z)
        return unormalized_log_w - torch.logsumexp(unormalized_log_w, dim=-1, keepdim=True)

class LocationScaleFlow(torch.nn.Module):
    def __init__(self, K, p):
        super().__init__()
        self.K = K
        self.p = p

        self.m = torch.nn.Parameter(torch.randn(self.K, self.p))
        self.log_s = torch.nn.Parameter(torch.zeros(self.K, self.p))

    def backward(self, z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)

    def forward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return (X-self.m.expand_as(X))/torch.exp(self.log_s).expand_as(X)

    def log_det_J(self,x):
        return -self.log_s.sum(-1)

class DIFDensityEstimator(torch.nn.Module):
    def __init__(self, target_samples, K,hidden_dims = [], estimate_reference_moments = False):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.K = K
        self.w = torch.distributions.Dirichlet(torch.ones(target_samples.shape[0])).sample()
        if estimate_reference_moments:
            self.reference_mean = torch.mean(self.target_samples,dim = 0)
            _ = torch.cov(self.target_samples.T)
            self.reference_cov = ((_.T + _)/2).reshape(self.p, self.p)
        else:
            self.reference_mean =torch.zeros(self.p)
            self.reference_cov = torch.eye(self.p)

        self.W = SoftmaxWeight(self.K, self.p, hidden_dims)

        self.T = LocationScaleFlow(self.K, self.p)
        if not estimate_reference_moments:
            self.T.m = torch.nn.Parameter(self.target_samples[torch.randint(low= 0, high = self.target_samples.shape[0],size = [self.K])])
            self.T.log_s = torch.nn.Parameter(torch.log(torch.var(self.target_samples, dim = 0).unsqueeze(0).repeat(self.K,1) + 1e-6 *torch.ones(self.K, self.p))/2)

    def initialize_with_EM(self, epochs, verbose = False):
        em = DiagGaussianMixtEM(self.target_samples,self.K)
        em.train(epochs, verbose)
        self.T.m = torch.nn.Parameter(em.m)
        self.T.log_s = torch.nn.Parameter(em.log_s)
        self.W.f[-1].weight = torch.nn.Parameter(torch.zeros(self.K,self.W.network_dimensions[-2]))
        self.W.f[-1].bias = torch.nn.Parameter(em.log_pi)
        self.reference_mean = torch.zeros(self.p)
        self.reference_cov = torch.eye(self.p)

    def reference_log_prob(self,z):
        return torch.distributions.MultivariateNormal(self.reference_mean.to(z.device), self.reference_cov.to(z.device)).log_prob(z)

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_log_v(self,x):
        z = self.T.forward(x)
        log_v = self.reference_log_prob(z) + torch.diagonal(self.W.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        z = self.T.forward(x)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x))).sample()
        return z[range(z.shape[0]), pick, :]

    def log_prob(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(self.reference_log_prob(z) + torch.diagonal(self.W.log_prob(z),0,-2,-1) + self.T.log_det_J(x),dim=-1)

    def sample(self, num_samples):
        z = torch.distributions.MultivariateNormal(self.reference_mean, self.reference_cov).sample(num_samples)
        x = self.T.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.W.log_prob(z))).sample()
        return x[range(x.shape[0]), pick, :]

    def M_step(self, x,w):
        v = torch.exp(self.compute_log_v(x))*w.unsqueeze(-1)
        c = torch.sum(v, dim=0)
        #self.log_pi = torch.log(c) - torch.logsumexp(torch.log(c), dim = 0)
        self.T.m = torch.nn.Parameter(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * x.unsqueeze(-2).repeat(1, self.K, 1),
                                dim=0) / c.unsqueeze(-1))
        temp = x.unsqueeze(1).repeat(1,self.K, 1) - self.T.m.unsqueeze(0).repeat(x.shape[0],1,1)
        temp2 = torch.square(temp)
        self.T.log_s = torch.nn.Parameter(torch.log(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * temp2,dim=0)/c.unsqueeze(-1))/2)

    def loss(self, x,w):
        return -torch.sum(w*self.log_prob(x))

    def train(self, epochs, batch_size = None, lr = 5e-3, weight_decay = 5e-5, verbose = False, trace_loss = False):
        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=lr, weight_decay=weight_decay)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        dataset = torch.utils.data.TensorDataset(self.target_samples.to(device), self.w.to(device))
        if trace_loss:
            loss_values = []
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                batch_loss = self.loss(batch[0],batch[1])
                batch_loss.backward()
                self.optimizer.step()
            if verbose or trace_loss:
                with torch.no_grad():
                    iteration_loss = torch.tensor([self.loss(batch[0],batch[1]) for i, batch in enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to(torch.device('cpu'))
        if trace_loss:
            return loss_values