import torch
from tqdm import tqdm
from conditional_density_estimation import *
from misc.metrics import *

class GenerativeClassifierSemiSupervised(torch.nn.Module):
    def __init__(self, samples, labels, structure, prior_probs=None):
        super().__init__()
        self.samples = samples
        self.sample_dim = samples.shape[-1]
        self.labels = labels
        self.C = labels.shape[-1]
        self.conditional_model = FlowConditionalDensityEstimation(samples, labels, structure)
        if prior_probs is None:
            self.prior_log_probs = torch.log(torch.ones(self.C) / self.C)
        else:
            self.prior_log_probs = torch.log(prior_probs)

    def to(self, device):
        for model in self.conditional_model.model:
            model.to(device)

    def log_prob(self, samples):
        augmented_samples = samples.unsqueeze(-2).repeat(1, self.C, 1).to(samples.device)
        augmented_labels = torch.eye(self.C).unsqueeze(0).repeat(samples.shape[0], 1, 1).to(samples.device)
        return self.conditional_model.log_prob(augmented_samples, augmented_labels)

    def loss(self, samples, labels):
        return -torch.sum(torch.sum(self.log_prob(samples) * labels, dim=-1))

    def train(self, epochs, batch_size=None, lr=5e-3, weight_decay=5e-5, unlabeled_samples=None, unlabeled_labels=None,
              test_samples=None, test_labels=None):
        self.conditional_model.initialize_with_EM(torch.cat([self.samples, unlabeled_samples], dim=0), 50)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        para_dict = []
        for model in self.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
        samples = torch.cat([self.samples, unlabeled_samples])
        labels = torch.cat([self.labels, torch.ones(unlabeled_samples.shape[0], self.C) / self.C])
        batch_size = samples.shape[0]
        dataset = torch.utils.data.TensorDataset(samples, labels)
        train_accuracy_trace = []
        unlabeled_accuracy_trace = []
        test_accuracy_trace = []
        pbar = tqdm(range(epochs))
        for __ in pbar:
            self.to(device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            if __ % 10 == 0:
                with torch.no_grad():
                    self.to(torch.device('cpu'))
                    iteration_loss = torch.tensor(
                        [self.loss(batch[0], batch[1]) for _, batch in enumerate(dataloader)]).sum().item()
                    train_accuracy = compute_accuracy(self.log_prob(self.samples), self.labels)
                    train_accuracy_trace.append(train_accuracy.item())
                    unlabeled_accuracy = compute_accuracy(self.log_prob(unlabeled_samples), unlabeled_labels)
                    unlabeled_accuracy_trace.append(unlabeled_accuracy.item())
                    test_accuracy = compute_accuracy(self.log_prob(test_samples), test_labels)
                    test_accuracy_trace.append(test_accuracy.item())
                    pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 4)) + '; device = ' + str(
                        device) + '; train_acc = ' + str(train_accuracy) + '; unlab_acc = ' + str(
                        unlabeled_accuracy) + '; test_acc= ' + str(test_accuracy))
        return train_accuracy_trace, unlabeled_accuracy_trace, test_accuracy_trace

class GenerativeClassifier(torch.nn.Module):
    def __init__(self,samples, labels, structure = [], prior_probs = None):
        super().__init__()
        self.samples = samples
        self.sample_dim = samples.shape[-1]
        self.labels = labels
        self.num_labels = labels.shape[-1]
        self.conditional_model = FlowConditionalDensityEstimation(samples, labels, structure=structure)
        if prior_probs is None:
            self.prior_log_probs = torch.log(torch.ones(self.num_labels)/self.num_labels)
        else:
            self.prior_log_probs = torch.log(prior_probs)

    def log_prob(self, samples):
        augmented_samples = samples.unsqueeze(-2).repeat(1,self.num_labels,1).to(samples.device)
        augmented_labels = torch.eye(self.num_labels).unsqueeze(0).repeat(samples.shape[0],1,1).to(samples.device)
        temp = self.conditional_model.log_prob(augmented_samples,augmented_labels) + self.prior_log_probs.unsqueeze(0).repeat(samples.shape[0],1).to(samples.device)
        return temp - torch.logsumexp(temp, dim = 1, keepdim= True)

    def loss(self, samples,labels,w):
        return -torch.sum(w*torch.sum(self.log_prob(samples)*labels, dim =-1))

    def loss_with_unlabeled(self, samples,labels,w, unlabeled_samples, unlabeled_w, prior):
        return -torch.sum(w*torch.sum(self.log_prob(samples)*labels, dim =-1)) - torch.sum(unlabeled_w*torch.sum(self.log_prob(unlabeled_samples)*prior, dim =-1))

    def train(self, epochs,batch_size=None, lr = 5e-3, weight_decay = 5e-5, verbose = False, unlabeled_samples = None, unlabeled_labels = None, test_samples = None, test_labels = None,trace_accuracy = False):
        w = torch.distributions.Dirichlet(torch.ones(self.samples.shape[0])).sample()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        if batch_size is None:
            batch_size = self.samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = torch.utils.data.TensorDataset(self.samples, self.labels, w)
        if trace_accuracy:
            train_accuracy_trace = []
            if (unlabeled_samples is not None) and (unlabeled_labels is not None):
                unlabeled_accuracy_trace = []
            else:
                unlabeled_accuracy_trace = None
            if (test_samples is not None) and (test_labels is not None):
                test_accuracy_trace = []
            else:
                test_accuracy_trace = None
        else:
            train_accuracy_trace = None
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            self.to(device)
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
                    if train_accuracy_trace is not None:
                        train_accuracy = compute_accuracy(self.cpu().log_prob(self.samples), self.labels)
                        train_accuracy_trace.append(train_accuracy.item())
                    else:
                        train_accuracy = None
                    if unlabeled_accuracy_trace is not None:
                        unlabeled_accuracy = compute_accuracy(self.cpu().log_prob(unlabeled_samples), unlabeled_labels)
                        unlabeled_accuracy_trace.append(unlabeled_accuracy.item())
                    else:
                        unlabeled_accuracy = None
                    if test_accuracy_trace is not None:
                        test_accuracy = compute_accuracy(self.cpu().log_prob(test_samples), test_labels)
                        test_accuracy_trace.append(test_accuracy.item())
                    else :
                        test_accuracy = None
                    pbar.set_postfix_str('loss = ' + str(round(iteration_loss,4)) + '; device = ' + str(device) + '; train_acc = ' + str(train_accuracy) + '; unlab_acc = ' + str(unlabeled_accuracy) +  '; test_acc= ' + str(test_accuracy))
        self.cpu()
        if trace_accuracy:
            return train_accuracy_trace,unlabeled_accuracy_trace, test_accuracy_trace


