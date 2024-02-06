import torch
from tqdm import tqdm
from .discretely_indexed_flows import SoftmaxWeight, LocationScaleFlow

class MaskedLinear(torch.nn.Linear):
    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        super().__init__(n_in, n_out, bias)
        self.mask = None

    def initialise_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)

class DIFDensityEstimationLayer(torch.nn.Module):
    def __init__(self,p, K,hidden_dims, q_log_prob):
        super().__init__()
        self.p = p
        self.K = K

        self.w = SoftmaxWeight(self.K, self.p, hidden_dims)
        self.T = LocationScaleFlow(self.K, self.p)

        self.q_log_prob = q_log_prob

    def log_v(self,x):
        with torch.no_grad():
            z = self.T.forward(x)
            log_v = self.q_log_prob(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
            return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_forward(self,x):
        with torch.no_grad():
            z = self.T.forward(x)
            pick = torch.distributions.Categorical(torch.exp(self.log_v(x))).sample()
            return torch.stack([z[i,pick[i],:] for i in range(z.shape[0])])

    def sample_backward(self, z):
        with torch.no_grad():
            x = self.T.backward(z)
            pick = torch.distributions.Categorical(torch.exp(self.w.log_prob(z))).sample()
            return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])])

    def log_prob(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(self.q_log_prob(z) + torch.diagonal(self.w.log_prob(z),0,-2,-1) + self.T.log_det_J(x),dim=-1)

class RealNVPDensityEstimationLayer(torch.nn.Module):
    def __init__(self,p,hidden_dim, q_log_prob):
        super().__init__()
        self.p = p
        net = []
        hs = [self.p] + hidden_dim + [2*self.p]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                torch.nn.Linear(h0, h1),
                torch.nn.Tanh(),
            ])
        net.pop()
        self.net = torch.nn.Sequential(*net)

        self.mask = [torch.cat([torch.zeros(int(self.p/2)), torch.ones(self.p - int(self.p/2))], dim = 0),torch.cat([torch.ones(int(self.p/2)), torch.zeros(self.p - int(self.p/2))], dim = 0)]
        self.q_log_prob = q_log_prob

    def sample_forward(self,x, return_log_det = True):
        z = x
        if return_log_det:
            log_det = torch.zeros(x.shape[:-1]).to(x.device)
        for mask in reversed(self.mask):
            out = self.net(mask * z)
            m, log_s = out[...,:self.p]*(1 - mask),out[...,self.p:]*(1 - mask)
            z = (z*(1 - mask) * torch.exp(log_s)+m) + (mask * z)
            if return_log_det:
                log_det += torch.sum(log_s, dim=-1)
        if return_log_det:
            return z, log_det
        else:
            return z

    def sample_backward(self, z):
        x = z
        for mask in self.mask:
            out = self.net(x*mask)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            x = ((x*(1-mask) -m)/torch.exp(log_s)) + (x*mask)
        return x

    def log_prob(self, x):
        z,log_det = self.sample_forward(x, return_log_det=True)
        return self.q_log_prob(z) + log_det

class MAFDensityEstimationLayer(torch.nn.Module):
    def __init__(self,p,hidden_dims, q_log_prob):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = p
        self.net = []
        self.hidden_dims=hidden_dims
        self.q_log_prob = q_log_prob
        self.lr = 5e-5

        hs = [self.p] + self.hidden_dims + [2*self.p]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                torch.nn.Tanh(),
            ])
        self.net.pop()
        self.net = torch.nn.Sequential(*self.net)

        self.m = {}
        self.update_masks()

    def update_masks(self):
        L = len(self.hidden_dims)
        self.m[-1] = torch.randperm(self.p)
        for l in range(L):
            self.m[l] = torch.randint(torch.min(self.m[l - 1]), self.p - 1, [self.hidden_dims[l]])

        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
        masks[-1] = masks[-1].repeat(1, 2)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m.to(self.device))


    def sample_forward(self,x, return_log_det = True):
        out = self.net(x)
        m, log_s = out[...,self.p:],out[...,:self.p]
        log_s = torch.clamp(log_s, max=10)
        if return_log_det:
            return m + torch.exp(log_s)*x, torch.sum(log_s, dim = -1)
        else:
            return m + torch.exp(log_s)*x

    def sample_backward(self,z):
        x = torch.zeros_like(z)
        for i in self.m[-1]:
            out = self.net(x)
            m, log_s = out[..., self.p:], out[..., :self.p]
            log_s = torch.clamp(log_s, max=10)
            temp = (z - m)/torch.exp(log_s)
            x[:,i] = temp[:,i]
        return x

    def log_prob(self, x):
        z,log_det = self.sample_forward(x, return_log_det=True)
        return self.q_log_prob(z) + log_det

class FlowDensityEstimation(torch.nn.Module):
    def __init__(self, target_samples,structure, estimate_reference = False):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.structure = structure
        self.N = len(self.structure)

        if estimate_reference:
            self.reference_mean = torch.mean(target_samples,dim = 0)
            _ = torch.cov(self.target_samples.T)
            self.reference_cov = ((_ + _.T)/2).reshape(self.p, self.p)
        else:
            self.reference_mean = torch.zeros(self.p)
            self.reference_cov = torch.eye(self.p)

        self.w = torch.distributions.Dirichlet(torch.ones(target_samples.shape[0])).sample()

        self.model = [structure[-1][0](self.p,self.structure[-1][1], q_log_prob= self.reference_log_prob)]
        for i in range(self.N - 2, -1, -1):
            self.model.insert(0, structure[i][0](self.p, structure[i][1], q_log_prob=self.model[0].log_prob))

    def reference_log_prob(self,z):
        return torch.distributions.MultivariateNormal(self.reference_mean.to(z.device), self.reference_cov.to(z.device)).log_prob(z)

    def compute_number_params(self):
        number_params = 0
        for model in self.model:
            number_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
        return number_params

    def sample(self, num_samples):
        z = torch.distributions.MultivariateNormal(self.reference_mean, self.reference_cov).sample(num_samples)
        for i in range(self.N - 1, -1, -1):
            z = self.model[i].sample_backward(z)
        return z

    def sample_latent(self, x):
        for i in range(self.N):
            x = self.model[i].sample_forward(x)
        return x

    def log_prob(self, x):
        return self.model[0].log_prob(x)

    def loss(self, x,w):
        return - torch.sum(w*self.log_prob(x))

    def train(self, epochs, batch_size = None, lr = 5e-3, weight_decay = 5e-5, verbose = False, trace_loss = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.para_dict = []
        for model in self.model:
            self.para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay':weight_decay})
            model.to(device)
        self.optimizer = torch.optim.Adam(self.para_dict)

        if trace_loss:
            loss_values = []
        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples.to(device), self.w.to(device))
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
                    iteration_loss = torch.tensor([self.loss(batch[0], batch[1]) for i, batch in
                                                   enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to('cpu')
        for layer in self.model:
            layer.to(torch.device('cpu'))
        if trace_loss:
            return loss_values

