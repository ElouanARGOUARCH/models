import torch
from tqdm import tqdm
from conditional_density_estimation import ConditionalDIF
from misc.metrics import *

class GenerativeClassifier(torch.nn.Module):
    def __init__(self,samples, labels, hidden_dims = [], prior_probs = None, K = 10):
        super().__init__()
        self.samples = samples
        self.p = samples.shape[-1]
        self.labels = labels
        self.K = labels.shape[-1]
        self.conditional_model = ConditionalDIF(samples, labels, K, hidden_dims)
        self.conditional_model.initialize_with_EM(10)
        if prior_probs is None:
            self.prior_log_probs = torch.log(torch.ones(self.K)/self.K)
        else:
            self.prior_log_probs = torch.log(prior_probs)
        self.w = torch.distributions.Dirichlet(torch.ones(samples.shape[0])).sample()


    def log_prob(self, samples):
        augmented_samples = samples.unsqueeze(-2).repeat(1,self.K,1).to(samples.device)
        augmented_labels = torch.eye(self.K).unsqueeze(0).repeat(samples.shape[0],1,1).to(samples.device)
        temp = self.conditional_model.log_prob(augmented_samples,augmented_labels) + self.prior_log_probs.unsqueeze(0).repeat(samples.shape[0],1).to(samples.device)
        return temp - torch.logsumexp(temp, dim = 1, keepdim= True)

    def loss(self, samples,labels,w):
        return -torch.sum(w*torch.sum(self.log_prob(samples)*labels, dim =-1))

    def train(self, epochs,batch_size=None, lr = 5e-3, weight_decay = 5e-5, verbose = False, test_samples = None, test_labels = None, trace_accuracy = False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        if batch_size is None:
            batch_size = self.samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = torch.utils.data.TensorDataset(self.samples, self.labels, self.w)
        if trace_accuracy:
            train_accuracy_trace = []
            if (test_samples is not None) and (test_labels is not None):
                test_accuracy_trace = []
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
                    train_accuracy = compute_accuracy(self.cpu().log_prob(self.samples), self.labels)
                    if trace_accuracy:
                        train_accuracy_trace.append(str(train_accuracy))
                if (test_samples is not None) and (test_labels is not None):
                    test_accuracy = compute_accuracy(self.log_prob(test_samples), test_labels)
                    if trace_accuracy:
                        test_accuracy_trace.append(str(test_accuracy))
                    pbar.set_postfix_str('loss = ' + str(round(iteration_loss,4)) + '; device = ' + str(device) + '; train_accuracy = ' + str(train_accuracy) + '; test_accuracy = ' + str(test_accuracy))
                else:
                    pbar.set_postfix_str('loss = ' + str(round(iteration_loss,4)) + '; device = ' + str(device) + '; train_accuracy = ' +str(train_accuracy))
        self.cpu()
        if trace_accuracy:
            if (test_samples is not None) and (test_labels is not None):
                return train_accuracy_trace, test_accuracy_trace
            else:
                return train_accuracy_trace


