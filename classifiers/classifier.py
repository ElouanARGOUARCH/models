import torch
from tqdm import tqdm

class BinaryClassifier(torch.nn.Module):
    def __init__(self, label_0_samples, label_1_samples,hidden_dims = [] ):
        super().__init__()
        self.label_0_samples = label_0_samples
        self.label_1_samples = label_1_samples
        assert label_0_samples.shape[-1]==label_1_samples.shape[-1],'mismatch in sample dimensions'
        self.p = label_0_samples.shape[-1]
        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.SiLU(), ])
        network.pop()
        self.logit_r = torch.nn.Sequential(*network)
        self.w = torch.distributions.Dirichlet(torch.ones(self.label_0_samples.shape[0] + self.label_1_samples.shape[0])).sample()

    def loss(self, samples, labels, w):
        label_0_samples = samples[labels == 0]
        label_0_w = w[labels == 0]
        label_1_samples = samples[labels == 1]
        label_1_w = w[labels == 1]
        log_sigmoid = torch.nn.LogSigmoid()
        return -torch.mean(log_sigmoid(self.logit_r(label_1_samples))) - torch.mean(log_sigmoid(-self.logit_r(label_0_samples)))

    def train(self, epochs, batch_size=None, lr=5e-3, weight_decay=5e-6):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if batch_size is None:
            batch_size = self.label_1_samples.shape[0] + self.label_0_samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        samples = torch.cat([self.label_0_samples, self.label_1_samples], dim = 0).to(device)
        labels= torch.cat([torch.zeros(self.label_0_samples.shape[0]), torch.ones(self.label_1_samples.shape[0])], dim = 0).long().to(device)
        dataset = torch.utils.data.TensorDataset(samples, labels, self.w.to(device))

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for batch in dataloader:
                optimizer.zero_grad()
                batch_loss = self.loss(batch[0], batch[1], batch[2])
                batch_loss.backward()
                optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0], batch[1],batch[2]) for batch in dataloader]).mean().item()
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 6)) + '; device = ' + str(device))
        self.to(torch.device('cpu'))

class Classifier(torch.nn.Module):
    def __init__(self, K, samples, labels, hidden_dimensions =[]):
        super().__init__()
        self.K = K
        assert samples.shape[0] == labels.shape[0],'number of samples does not match number of samples'
        self.samples = samples
        self.labels = labels
        self.p = samples.shape[-1]
        self.network_dimensions = [self.p] + hidden_dimensions + [self.K]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1),torch.nn.Tanh(),])
        self.f = torch.nn.Sequential(*network)

        self.w = torch.distributions.Dirichlet(torch.ones(samples.shape[0])).sample()

    def log_prob(self, samples):
        temp = self.f.forward(samples)
        return temp - torch.logsumexp(temp, dim = -1, keepdim=True)

    def loss(self, samples,labels,w):
        return -torch.sum(w*(self.log_prob(samples))[range(samples.shape[0]), labels])

    def train(self, epochs,batch_size=None, lr = 5e-3, weight_decay = 5e-5, verbose = False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        if batch_size is None:
            batch_size = self.samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        dataset = torch.utils.data.TensorDataset(self.samples, self.labels, self.w)

        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _,batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device), batch[2].to(device))
                loss.backward()
                optimizer.step()
            if verbose:
                with torch.no_grad():
                    iteration_loss = torch.tensor(
                        [self.loss(batch[0].to(device), batch[1].to(device), batch[2].to(device)) for _, batch in
                         enumerate(dataloader)]).sum().item()
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss,4)) + '; device = ' + str(device))
        self.cpu()
