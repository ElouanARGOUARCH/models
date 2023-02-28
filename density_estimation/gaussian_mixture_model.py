import torch
from tqdm import tqdm

class DiagGaussianMixtEM(torch.nn.Module):
    def __init__(self,target_samples,K):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.K = K
        self.log_pi = torch.log(torch.ones([self.K])/self.K)
        self.m = self.target_samples[torch.randint(low= 0, high = self.target_samples.shape[0],size = [self.K])]
        self.log_s = torch.log(torch.var(self.target_samples, dim = 0)).unsqueeze(0).repeat(self.K, 1)/2

        self.loss_values = []

    def forward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return (X - self.m.expand_as(X)) / torch.exp(self.log_s).expand_as(X)

    def backward(self,z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)

    def log_det_J(self,x):
        return -torch.sum(self.log_s, dim = -1)

    def compute_log_v(self,x):
        z = self.forward(x)
        unormalized_log_v = self.reference_log_density(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1)+ self.log_det_J(x)
        return unormalized_log_v - torch.logsumexp(unormalized_log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        z = self.forward(x)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x))).sample()
        return torch.stack([z[i,pick[i],:] for i in range(x.shape[0])])

    def sample_reference(self, num_samples):
        return torch.distributions.MultivariateNormal(torch.zeros(self.p), torch.eye(self.p)).sample(num_samples)

    def reference_log_density(self, z):
        return -torch.sum(torch.square(z)/2, dim = -1) - torch.log(torch.tensor([2*torch.pi], device = z.device))*self.p/2

    def log_density(self, x):
        z = self.forward(x)
        return torch.logsumexp(self.reference_log_density(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1) + self.log_det_J(x),dim=-1)

    def sample(self, num_samples):
        z = self.sample_reference(num_samples)
        x = self.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0],1))).sample()
        return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])])

    def M_step(self, batch):
        v = torch.exp(self.compute_log_v(batch))
        c = torch.sum(v, dim=0)
        self.log_pi = torch.log(c) - torch.logsumexp(torch.log(c), dim = 0)
        self.m = torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * batch.unsqueeze(-2).repeat(1, self.K, 1),
                                dim=0) / c.unsqueeze(-1)
        temp = batch.unsqueeze(1).repeat(1,self.K, 1) - self.m.unsqueeze(0).repeat(batch.shape[0],1,1)
        temp2 = torch.square(temp)
        self.log_s = torch.log(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * temp2,dim=0)/c.unsqueeze(-1))/2

    def train(self, epochs):
        pbar = tqdm(range(epochs))
        for t in pbar:
            self.M_step(self.target_samples)
            iteration_loss = -torch.mean(self.log_density(self.target_samples)).detach().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(iteration_loss))
    def sample_joint(self, num_samples):
        z = self.sample_reference(num_samples)
        x = self.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0],1))).sample()
        return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])]), pick

class FullRankGaussianMixtEM(torch.nn.Module):
    def __init__(self, target_samples, K):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.K = K
        self.log_pi = torch.log(torch.ones([self.K]) / self.K)
        self.m = self.target_samples[torch.randint(low=0, high=self.target_samples.shape[0], size=[self.K])]
        temp = torch.cov(self.target_samples.T)
        self.Sigma = ((temp + temp.T)/2).unsqueeze(0).repeat(self.K, 1, 1)

        self.loss_values = []

    def forward(self, x):
        X = x.unsqueeze(-2).repeat(1, self.K, 1)
        mean = self.m.unsqueeze(0).repeat(x.shape[0], 1, 1)
        delta = (X - mean).unsqueeze(-1)
        sqrt_inv_cov = torch.linalg.inv(torch.linalg.cholesky(self.Sigma)).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return torch.matmul(sqrt_inv_cov, delta).squeeze(-1)

    def backward(self, z):
        Z = z.unsqueeze(-2).repeat(1, self.K, 1)
        mean = self.m.unsqueeze(0).repeat(z.shape[0], 1, 1)
        sqrt_cov = torch.linalg.cholesky(self.Sigma).unsqueeze(0).repeat(z.shape[0], 1, 1, 1)
        return torch.matmul(sqrt_cov, Z.unsqueeze(-1)).squeeze(-1) + mean

    def log_det_J(self, x):
        return -torch.log(torch.linalg.det(self.Sigma)) / 2

    def compute_log_v(self, x):
        z = self.forward(x)
        unormalized_log_v = self.reference_log_prob(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1) + self.log_det_J(x)
        return unormalized_log_v - torch.logsumexp(unormalized_log_v, dim=-1, keepdim=True)

    def sample_latent(self, x):
        z = self.forward(x)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x))).sample()
        return torch.stack([z[i, pick[i], :] for i in range(x.shape[0])])

    def sample_reference(self, num_samples):
        return torch.distributions.MultivariateNormal(torch.zeros(self.p, torch.eye(self.p))).sample(num_samples)

    def reference_log_prob(self, z):
        return -torch.sum(torch.square(z) / 2, dim=-1) - torch.log(
            torch.tensor([2 * torch.pi], device=z.device)) * self.p / 2

    def log_prob(self, x):
        z = self.forward(x)
        return torch.logsumexp(
            self.reference_log_prob(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0], 1) + self.log_det_J(x), dim=-1)

    def sample(self, num_samples):
        z = self.sample_reference(num_samples)
        x = self.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0], 1))).sample()
        return torch.stack([x[i, pick[i], :] for i in range(z.shape[0])])

    def sample_joint(self, num_samples):
        z = self.sample_reference(num_samples)
        x = self.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0],1))).sample()
        return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])]), pick

    def M_step(self, batch):
        v = torch.exp(self.compute_log_v(batch))
        c = torch.sum(v, dim=0)
        self.log_pi = torch.log(c) - torch.logsumexp(torch.log(c), dim=0)
        self.m = torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * batch.unsqueeze(-2).repeat(1, self.K, 1),
                           dim=0) / c.unsqueeze(-1)
        temp = (batch.unsqueeze(1).repeat(1, self.K, 1) - self.m.unsqueeze(0).repeat(batch.shape[0], 1, 1)).unsqueeze(
            -1)
        self.Sigma = torch.sum(v.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.p, self.p) * (temp @ torch.transpose(temp, -1, -2)),
            dim=0) / (c.unsqueeze(-1).unsqueeze(-1).repeat(1, self.p, self.p))

    def train(self, epochs):
        pbar = tqdm(range(epochs))
        for t in pbar:
            self.M_step(self.target_samples)
            iteration_loss = -torch.mean(self.log_prob(self.target_samples)).detach().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(iteration_loss))