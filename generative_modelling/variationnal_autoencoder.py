import torch
from tqdm import tqdm
class VAE(torch.nn.Module):
    def __init__(self, target_samples,latent_dim, hidden_sizes_encoder, hidden_sizes_decoder):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_samples = target_samples.to(self.device)
        self.p = target_samples.shape[1]
        self.d = latent_dim
        self.hidden_sizes_encoder = hidden_sizes_encoder
        self.hidden_sizes_decoder = hidden_sizes_decoder

        self.encoder = []
        hs = [self.p] + hidden_sizes_encoder + [2 * self.d]
        for h0, h1 in zip(hs, hs[1:]):
            self.encoder.extend([
                torch.nn.Linear(h0, h1),
                torch.nn.SiLU(),
            ])
        self.encoder.pop()
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.decoder = []
        hs = [self.d] + hidden_sizes_decoder + [2 * self.p]
        for h0, h1 in zip(hs, hs[1:]):
            self.decoder.extend([
                torch.nn.Linear(h0, h1),
                torch.nn.SiLU(),
            ])
        self.decoder.pop()
        self.decoder = torch.nn.Sequential(*self.decoder)
        self.prior_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d).to(self.device), torch.eye(self.d).to(self.device))

    def sample_encoder(self, x):
        out = self.encoder(x)
        mu, log_sigma = out[..., :self.d], out[..., self.d:]
        return mu + torch.exp(log_sigma) * torch.randn(list(x.shape)[:-1] + [self.d]).to(self.device)

    def encoder_log_density(self, z, x):
        out = self.encoder(x)
        mu, log_sigma = out[..., :self.d], out[..., self.d:]
        return -torch.sum(torch.square((z - mu)/torch.exp(log_sigma))/2 + log_sigma + self.d*torch.log(torch.tensor([2*3.14159265]).to(self.device))/2, dim=-1)

    def sample_decoder(self, z):
        out = self.decoder(z)
        mu, log_sigma = out[..., :self.p], out[..., self.p:]
        return mu + torch.exp(log_sigma) * torch.randn(list(z.shape)[:-1] + [self.p]).to(self.device)

    def DKL_posterior_prior(self,mu, log_sigma):
        return torch.sum(torch.square(mu) + torch.exp(log_sigma)+log_sigma-self.d, dim = -1)/2

    def decoder_log_density(self, x, z):
        out = self.decoder(z)
        mu, log_sigma = out[..., :self.p], out[..., self.p:]
        return -torch.sum(torch.square((x - mu)/torch.exp(log_sigma))/2 + log_sigma + self.p*torch.log(torch.tensor([2*3.14159265]).to(self.device))/2, dim=-1)

    def sample(self, num_samples):
        z = self.prior_distribution.sample(num_samples)
        return self.sample_decoder(z)

    def sample_latent(self, x):
        return self.sample_encoder(x)

    def resample_input(self, x):
        z = self.sample_proxy(x)
        return self.sample_model(z)

    def model_log_density(self, x, mc_samples = 100):
        x = x.unsqueeze(0).repeat(mc_samples, 1, 1)
        z = self.sample_encoder(x)
        return torch.logsumexp(self.decoder_log_density(x,z) + self.prior.log_prob(z) - self.encoder_log_density(z,x) - torch.log(torch.tensor([mc_samples]).to(self.device)), dim = 0)

    def ELBO(self, x):
        out = self.encoder(x)
        mu, log_sigma = out[..., :self.d], out[..., self.d:]
        DKL_values = self.DKL_posterior_prior(mu, log_sigma)
        MC_samples = 5
        x = x.unsqueeze(0).repeat(MC_samples, 1, 1)
        z = self.sample_encoder(x)
        mean_log_ratio = torch.mean(self.decoder_log_density(x,z) - self.encoder_log_density(z,x), dim = 0)
        return  mean_log_ratio - DKL_values

    def loss(self, batch):
        return -torch.mean(self.ELBO(batch))

    def train(self, epochs, batch_size=None, lr = 5e-3, weight_decay = 5e-6, verbose = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.encoder.parameters()), lr = lr, weight_decay=weight_decay)
        dataset = torch.utils.data.TensorDataset(self.target_samples)

        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                batch_loss = self.loss(batch[0].to(device))
                batch_loss.backward()
                self.optimizer.step()
            if verbose:
                with torch.no_grad():
                    iteration_loss = torch.tensor([self.loss(batch[0].to(device)) for i, batch in
                                                   enumerate(dataloader)]).sum().item()
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(device))
        self.to('cpu')