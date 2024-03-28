import torch
from tqdm import tqdm
from conditional_density_estimation import *
from misc.metrics import *

class GaussianClassifier:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.sample_dim = samples.shape[-1]
        self.C = labels.shape[-1]
        self.mu_0 = torch.zeros(self.sample_dim)
        self.Psi_0 = torch.eye(self.sample_dim)
        self.lbda = 1
        self.nu = 10

    def sample_theta_from_posterior(self, num_samples, samples=None, labels=None):
        list_means = []
        list_Sigmas = []
        if samples is None:
            samples = self.samples
        if labels is None:
            labels = self.labels
        for c in range(self.C):
            samples_c = samples[labels[:, c] == 1]
            N_c = samples_c.shape[0]
            lbda_N_c = self.lbda + N_c
            nu_N_c = self.nu + N_c
            empirical_mean_c = torch.mean(samples_c, dim=0)
            mu_N_c = (self.lbda * self.mu_0 + N_c * empirical_mean_c) / lbda_N_c
            S_c = torch.cov(samples_c.T) * (N_c - 1)
            temp = (empirical_mean_c - self.mu_0).unsqueeze(-1)
            Psi_N_c = self.Psi_0 + S_c + (self.lbda * N_c * temp @ temp.T) / (lbda_N_c)
            Sigma_c = torch.inverse(torch.distributions.Wishart(nu_N_c, torch.inverse(Psi_N_c)).sample(num_samples))
            means_c = torch.distributions.MultivariateNormal(mu_N_c, Sigma_c / lbda_N_c).sample()
            list_means.append(means_c.unsqueeze(1))
            list_Sigmas.append(Sigma_c.unsqueeze(1))
        return torch.cat(list_means, dim=1), torch.cat(list_Sigmas, dim=1)

    def log_prob(self, samples, means, Sigmas, prior):
        mvn = torch.distributions.MultivariateNormal(means, Sigmas).log_prob(
            samples.unsqueeze(1).unsqueeze(1).repeat(1, means.shape[0], self.C, 1)) + torch.log(prior)
        return mvn - torch.logsumexp(mvn, dim=2, keepdim=True)

    def predict_with_gibbs(self, obs, prior, number_steps=500, samples=None, labels=None):
        means, Sigmas = self.sample_theta_from_posterior([1])
        for t in tqdm(range(number_steps)):
            log_prob = self.log_prob(obs, means, Sigmas, prior)[:, 0, :]
            labels = torch.distributions.Categorical(torch.exp(log_prob)).sample()
            augmented_labels = torch.cat([torch.nn.functional.one_hot(labels, num_classes=self.C), self.labels], dim=0)
            augmented_samples = torch.cat([obs, self.samples], dim=0)
            means, Sigmas = self.sample_theta_from_posterior([1], augmented_samples, augmented_labels)
        return log_prob

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
        return -torch.sum(self.log_prob(samples)* labels)

    def train(self, epochs, batch_size=None, unlabeled_samples=None, unlabeled_labels=None,
              test_samples=None, test_labels=None, recording_frequency = 1, lr=5e-3, weight_decay=5e-5):
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
            if __ % recording_frequency == 0 or __<100:
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
        return -torch.sum(self.conditional_model.log_prob(samples, labels))

    def train(self, epochs, batch_size=None, unlabeled_samples=None, unlabeled_labels=None,
              test_samples=None, test_labels=None, recording_frequency = 1, lr=5e-3, weight_decay=5e-5):
        self.conditional_model.initialize_with_EM(self.samples, 50, verbose=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        para_dict = []
        for model in self.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
        dataset = torch.utils.data.TensorDataset(self.samples, self.labels)
        train_accuracy_trace = []
        unlabeled_accuracy_trace = []
        test_accuracy_trace = []
        indices = []
        pbar = tqdm(range(epochs))
        for __ in pbar:
            self.to(device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            if __ % recording_frequency == 0 or __<100:
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
                    indices.append(__)
                    pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 4)) + '; device = ' + str(
                        device) + '; train_acc = ' + str(train_accuracy) + '; unlab_acc = ' + str(
                        unlabeled_accuracy) + '; test_acc= ' + str(test_accuracy))
        return train_accuracy_trace, unlabeled_accuracy_trace, test_accuracy_trace, indices
