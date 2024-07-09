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

    def compute_number_params(self):
        return self.conditional_model.compute_number_params()

    def to(self, device):
        for model in self.conditional_model.model:
            model.to(device)

    def log_prob(self, samples):
        augmented_samples = samples.unsqueeze(-2).repeat(1, self.C, 1).to(samples.device)
        augmented_labels = torch.eye(self.C).unsqueeze(0).repeat(samples.shape[0], 1, 1).to(samples.device)
        return self.conditional_model.log_prob(augmented_samples, augmented_labels)

    def log_posterior_prob(self, samples, prior):
        return torch.softmax(self.log_prob(samples) + torch.log(prior.unsqueeze(0)), dim=-1)

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
    def __init__(self, samples_dim, labels_dim, structure, prior_probs=None):
        super().__init__()
        self.sample_dim = samples_dim
        self.C = labels_dim
        self.structure = structure
        self.conditional_model = FlowConditionalDensityEstimation(torch.randn(1, samples_dim),
                                                                  torch.ones(1, labels_dim), structure)
        if prior_probs is None:
            self.prior_log_probs = torch.log(torch.ones(self.C) / self.C)
        else:
            self.prior_log_probs = torch.log(prior_probs)

    def compute_number_params(self):
        return self.conditional_model.compute_number_params()

    def to(self, device):
        for model in self.conditional_model.model:
            model.to(device)

    def log_prob(self, samples):
        augmented_samples = samples.unsqueeze(-2).repeat(1, self.C, 1).to(samples.device)
        augmented_labels = torch.eye(self.C).unsqueeze(0).repeat(samples.shape[0], 1, 1).to(samples.device)
        return self.conditional_model.log_prob(augmented_samples, augmented_labels)

    def loss(self, samples, labels):
        return -torch.mean(torch.sum(self.log_prob(samples)*labels, dim = -1), dim = 0)

    def log_posterior_prob(self, samples, prior):
        log_joint = self.log_prob(samples) + torch.log(prior.unsqueeze(0))
        return log_joint - torch.logsumexp(log_joint, dim = -1, keepdim=True)

    def train(self, epochs, batch_size, train_samples, train_labels, list_test_samples = [], list_test_prior_probs = [],list_test_labels = [],verbose = False, recording_frequency=1, lr=5e-3, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        para_dict = []
        for model in self.conditional_model.model:
            para_dict.insert(-1, {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        optimizer = torch.optim.Adam(para_dict)
        total_samples = torch.cat([train_samples] + list_test_samples, dim = 0)
        total_labels = torch.cat([train_labels] + [list_test_prior_probs[i].unsqueeze(0).repeat(list_test_samples[i].shape[0],1) for i in range(len(list_test_prior_probs))], dim=0)
        dataset = torch.utils.data.TensorDataset(total_samples, total_labels)
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
                        postfix_str += '; test_loss_' + str(i) + ' = ' + str(round(test_loss, 4))
                    pbar.set_postfix_str(postfix_str)
        self.to(torch.device('cpu'))
        if verbose:
            return train_loss_trace, list_test_loss_trace

    def gibbs(self, T, epochs, batch_size, train_samples,train_prior_probs, train_labels,list_test_samples = [], list_test_prior_probs = [], list_test_labels = [], recording_frequency = 1, lr = 5e-3, weight_decay = 5e-5):
        self.train(epochs, batch_size, train_samples, train_labels, [],[],[],False,recording_frequency, lr, weight_decay)
        total_samples = torch.cat([train_samples] + list_test_samples, dim = 0)
        print(compute_accuracy(self.log_posterior_prob(train_samples, train_prior_probs), train_labels))
        total_labels = [train_labels]
        for i in range(len(list_test_samples)):
            print(compute_accuracy(self.log_posterior_prob(list_test_samples[i], list_test_prior_probs[i]),list_test_labels[i]))
            total_labels += [torch.nn.functional.one_hot(torch.distributions.Categorical(torch.exp(self.log_posterior_prob(list_test_samples[i], list_test_prior_probs[i]))).sample(),num_classes=self.C)]
        total_labels = torch.cat(total_labels, dim=0)
        for t in range(T):
            self.conditional_model = FlowConditionalDensityEstimation(torch.randn(1, self.sample_dim),torch.ones(1, self.C), self.structure)
            self.train(epochs, batch_size, total_samples, total_labels, [],[],[],False,recording_frequency, lr, weight_decay)
            print(compute_accuracy(self.log_posterior_prob(train_samples, train_prior_probs), train_labels))
            total_labels = [train_labels]
            for i in range(len(list_test_samples)):
                print(compute_accuracy(self.log_posterior_prob(list_test_samples[i], list_test_prior_probs[i]), list_test_labels[i]))
                total_labels += [torch.nn.functional.one_hot(torch.distributions.Categorical(torch.exp(self.log_posterior_prob(list_test_samples[i], list_test_prior_probs[i]))).sample(), num_classes=self.C)]
            total_labels = torch.cat(total_labels, dim=0)