import torch
from tqdm import tqdm
from density_estimation import SoftmaxWeight, DiagGaussianMixtEM
import math

class ConditionalRealNVPLayer(torch.nn.Module):
    def __init__(self, sample_dim, label_dim, reference_log_prob, **kwargs):
        super().__init__()
        self.sample_dim = sample_dim
        self.label_dim = label_dim
        net = []
        hs = [self.sample_dim + self.label_dim] + kwargs['hidden_dims'] + [2 * self.sample_dim]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                torch.nn.Linear(h0, h1),
                torch.nn.Tanh(),
            ])
        net.pop()
        self.net = torch.nn.Sequential(*net)

        self.mask = [
            torch.cat([torch.zeros(int(self.sample_dim / 2)), torch.ones(self.sample_dim - int(self.sample_dim / 2))],
                      dim=0),
            torch.cat([torch.ones(int(self.sample_dim / 2)), torch.zeros(self.sample_dim - int(self.sample_dim / 2))],
                      dim=0)]
        self.reference_log_prob = reference_log_prob

    def sample_forward(self, samples, labels, return_log_det=True):
        latents = samples
        if return_log_det:
            log_det = torch.zeros(samples.shape[:-1]).to(samples.device)
        for mask in reversed(self.mask):
            mask = mask.to(samples.device)
            m, log_s = torch.chunk(self.net(torch.cat([mask * latents, labels], dim=-1)), 2, dim=-1)
            latents = (latents*torch.exp(log_s) + m)*(1-mask) + (mask * latents)
            if return_log_det:
                log_det += torch.sum(log_s * (1 - mask), dim=-1)
        if return_log_det:
            return latents, log_det
        else:
            return latents

    def sample_backward(self, latents, labels):
        samples = latents
        for mask in self.mask:
            mask = mask.to(latents.device)
            m, log_s = torch.chunk(self.net(torch.cat([samples * mask, labels], dim=-1)), 2, dim=-1)
            samples = ((samples - m) / torch.exp(log_s))*(1-mask) + (samples * mask)
        return samples

    def log_prob(self, samples, labels):
        latents, log_det = self.sample_forward(samples, labels, return_log_det=True)
        return self.reference_log_prob(latents, labels) + log_det

class ConditionalLocationScale(torch.nn.Module):
    def __init__(self, K, sample_dim, label_dim, hidden_dims):
        super().__init__()
        self.K = K
        self.sample_dim = sample_dim
        self.label_dim = label_dim

        self.network_dimensions = [self.label_dim] + hidden_dims + [2 * self.K * self.sample_dim]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.Tanh(), ])
        network.pop()
        self.f = torch.nn.Sequential(*network)

    def backward(self, latent, label, return_log_det=False):
        assert latent.shape[:-1] == label.shape[
                                    :-1], 'number of latent sample does not match the number of label sample'
        desired_size = list(latent.shape)
        desired_size.insert(-1, self.K)
        Latent = latent.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2 * self.sample_dim
        out = torch.reshape(self.f(label), new_desired_size)
        m, log_s = torch.chunk(out, 2, dim=-1)
        if return_log_det:
            return Latent * torch.exp(log_s).expand_as(Latent) + m.expand_as(Latent), -log_s.sum(-1)
        else:
            return Latent * torch.exp(log_s).expand_as(Latent) + m.expand_as(Latent)

    def forward(self, sample, label, return_log_det=False):
        assert sample.shape[:-1] == label.shape[:-1], 'number of x sample does not match the number of label sample'
        desired_size = list(sample.shape)
        desired_size.insert(-1, self.K)
        Sample = sample.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2 * self.sample_dim
        m, log_s = torch.chunk(torch.reshape(self.f(label), new_desired_size), 2, dim=-1)
        if return_log_det:
            return (Sample - m.expand_as(Sample)) / torch.exp(log_s).expand_as(Sample), -log_s.sum(-1)
        else:
            return (Sample - m.expand_as(Sample)) / torch.exp(log_s).expand_as(Sample)

class ConditionalDIFLayer(torch.nn.Module):
    def __init__(self, sample_dim, label_dim, reference_log_prob=None, **kwargs):
        super().__init__()
        self.sample_dim = sample_dim
        self.label_dim = label_dim
        self.K = kwargs['K']

        self.W = SoftmaxWeight(self.K, self.sample_dim + self.label_dim, kwargs['hidden_dims'])

        self.T = ConditionalLocationScale(self.K, self.sample_dim, self.label_dim, kwargs['hidden_dims'])

        self.reference_log_prob = reference_log_prob

    def initialize_with_EM(self, samples, epochs, verbose=False):
        em = DiagGaussianMixtEM(samples, self.K)
        em.train(epochs, verbose)
        self.T.f[-1].weight = torch.nn.Parameter(
            torch.zeros(self.T.network_dimensions[-1], self.T.network_dimensions[-2]))
        self.T.f[-1].bias = torch.nn.Parameter(torch.cat([em.m, em.log_s], dim=-1).flatten())
        self.W.f[-1].weight = torch.nn.Parameter(
            torch.zeros(self.W.network_dimensions[-1], self.W.network_dimensions[-2]))
        self.W.f[-1].bias = torch.nn.Parameter(em.log_pi)
        self.reference_mean = torch.zeros(self.sample_dim)
        self.reference_cov = torch.eye(self.sample_dim)

    def compute_log_v(self, sample, label):
        assert sample.shape[:-1] == label.shape[:-1], 'wrong shapes'
        label_unsqueezed = label.unsqueeze(-2).repeat(1, self.K, 1)
        latent, log_det = self.T.forward(sample, label, return_log_det=True)
        log_v = self.reference.log_prob(latent) + torch.diagonal(
            self.W.log_prob(torch.cat([latent, label_unsqueezed], dim=-1)),
            0, -2, -1) + log_det
        return log_v - torch.logsumexp(log_v, dim=-1, keepdim=True)

    def sample_forward(self, sample, label):
        assert sample.shape[:-1] == label.shape[:-1], 'wrong shapes'
        latent = self.T.forward(sample, label)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(sample, label))).sample()
        return latent[range(latent.shape[0]), pick, :]

    def log_prob(self, sample, label):
        assert sample.shape[:-1] == label.shape[:-1], 'wrong shapes'
        desired_size = list(label.shape)
        desired_size.insert(-1, self.K)
        label_unsqueezed = label.unsqueeze(-2).expand(desired_size)
        latent, log_det = self.T.forward(sample, label, return_log_det=True)
        return torch.logsumexp(self.reference_log_prob(latent, label_unsqueezed) + torch.diagonal(
            self.W.log_prob(torch.cat([latent, label_unsqueezed], dim=-1)), 0, -2, -1) + log_det, dim=-1)

    def sample_backward(self, latent, label):
        sample = self.T.backward(latent, label)
        pick = torch.distributions.Categorical(torch.exp(self.W.log_prob(torch.cat([latent, label], dim=-1)))).sample()
        return sample[range(sample.shape[0]), pick, :]

class FlowConditionalDensityEstimation(torch.nn.Module):
    def __init__(self, samples, labels, structure):
        super().__init__()
        self.samples = samples
        self.labels = labels
        self.sample_dim = samples.shape[-1]
        self.label_dim = labels.shape[-1]
        self.structure = structure
        self.N = len(self.structure)

        self.reference_mean = torch.zeros(self.sample_dim)
        self.reference_cov = torch.eye(self.sample_dim)

        self.model = [
            structure[-1][0](self.sample_dim, self.label_dim, self.reference_log_prob, **self.structure[-1][1])]
        for i in range(self.N - 2, -1, -1):
            self.model.insert(0, structure[i][0](self.sample_dim, self.label_dim, self.model[0].log_prob,
                                                 **structure[i][1]))

    def initialize_with_EM(self, samples, epochs, verbose=True):
        for model in self.model:
            if isinstance(model, ConditionalDIFLayer):
                model.initialize_with_EM(samples, epochs, verbose)
                break

    def reference_log_prob(self, latents, labels):
        return -latents.shape[-1] * torch.log(torch.tensor(2 * math.pi)) / 2 - torch.sum(torch.square(latents), dim=-1)/2

    def compute_number_params(self):
        number_params = 0
        for model in self.model:
            number_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
        return number_params

    def sample(self, labels):
        latents = torch.distributions.MultivariateNormal(self.reference_mean, self.reference_cov).sample(
            [labels.shape[0]])
        for i in range(self.N - 1, -1, -1):
            latents = self.model[i].sample_backward(latents, labels)
        return latents

    def sample_latent(self, samples, labels):
        assert samples.shape[0] == labels.shape[0], 'mismatch number samples'
        for i in range(self.N):
            samples = self.model[i].sample_forward(samples, labels)
        return samples

    def log_prob(self, samples, labels):
        assert samples.shape[0] == labels.shape[0], 'mismatch number samples'
        return self.model[0].log_prob(samples, labels)

    def loss(self, samples, labels):
        return - torch.sum(self.log_prob(samples, labels))

    def train(self, epochs, batch_size=None, lr=5e-3, weight_decay=5e-5, verbose=False, trace_loss=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.para_dict = []
        for model in self.model:
            self.para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
            model.to(device)
        self.optimizer = torch.optim.Adam(self.para_dict)

        if trace_loss:
            loss_values = []
        if batch_size is None:
            batch_size = self.samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.samples.to(device), self.labels.to(device))
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                batch_loss = self.loss(batch[0], batch[1])
                batch_loss.backward()
                self.optimizer.step()
            if verbose or trace_loss:
                with torch.no_grad():
                    iteration_loss = torch.tensor([self.loss(batch[0], batch[1]) for i, batch in
                                                   enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 6)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to('cpu')
        for layer in self.model:
            layer.to(torch.device('cpu'))
        if trace_loss:
            return loss_values

