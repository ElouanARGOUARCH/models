import torch
from tqdm import tqdm
from misc.metrics import *

class BinaryClassifier(torch.nn.Module):
    def __init__(self, label_0_samples, label_1_samples,hidden_dims = []):
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

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def loss(self, samples, labels, w):
        label_0_samples = samples[labels == 0]
        label_0_w = w[labels == 0]
        label_1_samples = samples[labels == 1]
        label_1_w = w[labels == 1]
        log_sigmoid = torch.nn.LogSigmoid()
        return -torch.sum(label_1_w * log_sigmoid(self.logit_r(label_1_samples))) - torch.sum(label_0_w * log_sigmoid(-self.logit_r(label_0_samples)))

    def train(self, epochs, batch_size=None, lr=5e-3, weight_decay=5e-6, verbose = False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if batch_size is None:
            batch_size = self.label_1_samples.shape[0] + self.label_0_samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        samples = torch.cat([self.label_0_samples, self.label_1_samples], dim = 0).to(device)
        labels= torch.cat([torch.zeros(self.label_0_samples.shape[0]), torch.ones(self.label_1_samples.shape[0])], dim = 0).long().to(device)
        dataset = torch.utils.data.TensorDataset(samples, labels, self.w.to(device))

        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
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
    def __init__(self, sample_dim, C, hidden_dimensions=[]):
        super().__init__()
        self.sample_dim = sample_dim
        self.C = C
        self.network_dimensions = [self.sample_dim] + hidden_dimensions + [self.C]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.Tanh(), ])
        self.f = torch.nn.Sequential(*network)

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def log_prob(self, samples):
        temp = self.f.forward(samples)
        return temp - torch.logsumexp(temp, dim=-1, keepdim=True)

    def loss(self, samples, labels):
        return -torch.mean(self.log_prob(samples) * labels)

    def train(self, epochs, batch_size,train_samples, train_labels,list_test_samples = [], list_test_labels = [],verbose = False, recording_frequency = 1, lr=5e-4, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters())
        dataset = torch.utils.data.TensorDataset(train_samples, train_labels)
        if verbose:
            train_loss_trace = []
            list_test_loss_trace = [[] for i in range(len(list_test_samples))]
        pbar = tqdm(range(epochs))
        for __ in pbar:
            self.to(device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            if __ % recording_frequency == 0 and verbose:
                with torch.no_grad():
                    self.to(torch.device('cpu'))
                    train_loss = self.loss(train_samples, train_labels).item()
                    train_loss_trace.append(train_loss)
                    postfix_str = 'device = ' + str(
                        device) + '; train_loss = ' + str(round(train_loss, 4))
                    for i in range(len(list_test_samples)):
                        test_loss = self.loss(list_test_samples[i], list_test_labels[i]).item()
                        list_test_loss_trace[i].append(test_loss)
                        postfix_str += '; test_loss_'+ str(i) +' = ' + str(round(test_loss, 4))
                    pbar.set_postfix_str(postfix_str)
        self.to(torch.device('cpu'))
        if verbose:
            return train_loss_trace, list_test_loss_trace