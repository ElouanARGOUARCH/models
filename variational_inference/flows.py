import torch
from tqdm import tqdm

class RealNVPSamplerLayer(torch.nn.Module):
    def __init__(self,p,hidden_dim, p_log_prob):
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
        self.q_log_prob = None
        self.p_log_prob = p_log_prob
        self.lr = 5e-3
        self.weight_decay = 5e-5

    def latent_log_prob(self,z):
        x = z
        log_det = torch.zeros(z.shape[:-1]).to(z.device)
        for mask in self.mask:
            mask = mask.to(z.device)
            out = self.net(x*mask)
            m, log_s = out[...,:self.p]*(1 - mask),out[...,self.p:]* (1 - mask)
            x = (x*(1-mask) -m)/torch.exp(log_s) + x*mask
            log_det -= torch.sum(log_s, dim=-1)
        return self.p_log_prob(x) + log_det

    def sample_backward(self, z):
        x = z
        for mask in self.mask:
            out = self.net(x*mask)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            x = (x*(1-mask) -m)/torch.exp(log_s) + x*mask
        return x

    def log_prob(self, x):
        z = x
        log_det = torch.zeros(x.shape[:-1])
        for mask in reversed(self.mask):
            out = self.net(mask * z)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            z = z*(1 - mask) * torch.exp(log_s) + m + mask * z
            log_det += torch.sum(log_s, dim = -1)
        return self.q_log_prob(z) + log_det

class FlowSampler(torch.nn.Module):
    def __init__(self, target_log_prob, p, structure):

        super().__init__()
        self.target_log_prob = target_log_prob
        self.p = p
        self.structure = structure
        self.N = len(self.structure)
        self.reference = torch.distributions.MultivariateNormal(torch.zeros(self.p), torch.eye(self.p))

        self.model = [structure[0][0](self.p, self.structure[0][1], p_log_prob=self.target_log_prob)]
        for i in range(1, self.N):
            self.model.append(structure[i][0](self.p, structure[i][1], p_log_prob=self.model[i - 1].latent_log_prob))
        for i in range(self.N - 1):
            self.model[i].q_log_prob = self.model[i + 1].log_prob
        self.model[-1].q_log_prob = self.reference.log_prob

    def compute_number_params(self):
        number_params = 0
        for model in self.model:
            number_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
        return number_params

    def sample(self, num_samples):
        z = self.reference.sample(num_samples)
        for i in range(self.N - 1, -1, -1):
            z = self.model[i].sample_backward(z)
        return z

    def log_prob(self, x):
        return self.model[0].log_prob(x)

    def latent_log_prob(self, z):
        return self.model[-1].latent_log_prob(z)

    def loss(self, batch):
        return - self.latent_log_prob(batch).mean()

    def DKL_latent(self, batch_z):
        return (self.reference.log_prob(batch_z) - self.latent_log_prob(batch_z)).mean()

    def DKL_observed(self, batch_x):
        return (self.log_prob(batch_x) - self.target_log_prob(batch_x)).mean()

    def train(self,epochs, num_samples,lr = 5e-3, weight_decay = 5e-5, verbose = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.para_dict = []
        for model in self.model:
            self.para_dict.insert(-1, {'params': model.parameters(), 'lr': lr,
                                       'weight_decay': weight_decay})
            model.to(device)
        self.optimizer = torch.optim.Adam(self.para_dict)
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            z = self.reference.sample([num_samples]).to(device)
            self.optimizer.zero_grad()
            loss = self.loss(z)
            loss.backward()
            self.optimizer.step()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(loss.item(), 6)) + ' ; device: ' + str(device))

        for model in self.model:
            model.to(torch.device('cpu'))
        self.to(torch.device('cpu'))