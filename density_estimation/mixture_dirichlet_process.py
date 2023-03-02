import pyro
import matplotlib
import matplotlib.pyplot as plt
from utils.visual import *
from tqdm import tqdm

class MixtureDirichletProcessMarginal(torch.nn.Module):
    def __init__(self, target_samples, nu = torch.tensor([3.]), lbda = torch.tensor([0.1]) ,alpha = torch.tensor([10.])):
        super().__init__()
        self.target_samples = target_samples
        self.d = target_samples.shape[-1]
        self.nu = nu
        self.lbda = lbda
        self.mu = torch.mean(target_samples,dim =0)
        self.psi = torch.eye(self.d)
        self.alpha = alpha
        self.z = torch.zeros(target_samples.shape[0])

    def compute_posterior_parameters(self,x):
        empirical_mean = torch.mean(x, dim =0)
        mu_N = (self.lbda*self.mu + x.shape[0]*empirical_mean)/(self.lbda + x.shape[0])
        S = torch.cov(x.T)*(x.shape[0]-1) if x.shape[0]>=2 else torch.zeros(self.d)
        temp = (empirical_mean-self.mu).unsqueeze(-1)
        psi_N = self.psi + S + (self.lbda*x.shape[0]*temp@temp.T)/(self.lbda + x.shape[0])
        return self.nu + x.shape[0], self.lbda + x.shape[0], mu_N, psi_N

    def compute_probability(self,i):
        z_i = torch.cat([self.z[:i], self.z[i+1:]], dim =0)
        list_weight = []
        list_evaluated_prob=[]
        for c in torch.unique(z_i):
            x_i = self.target_samples[self.z==c]
            nu_n_c, lbda_n_c, mu_n_c, psi_n_c = self.compute_posterior_parameters(x_i)
            list_weight.append(x_i.shape[0]/(self.target_samples.shape[0]-1+self.alpha))
            list_evaluated_prob.append(torch.exp(pyro.distributions.MultivariateStudentT(nu_n_c-self.d+1,mu_n_c,torch.cholesky(psi_n_c*(lbda_n_c+1)/(lbda_n_c*(nu_n_c - self.d + 1)))).log_prob(self.target_samples[i,:])))
        list_evaluated_prob.append(torch.exp(pyro.distributions.MultivariateStudentT(self.nu-self.d+1,self.mu,torch.cholesky(self.psi*(self.lbda+1)/(self.lbda*(self.nu - self.d + 1)))).log_prob(self.target_samples[i,:])))
        list_weight.append(self.alpha/(self.target_samples.shape[0]-1+self.alpha))
        probs = torch.tensor(list_weight)*torch.tensor(list_evaluated_prob)
        return probs

    def log_prob(self,x):
        list_weight = []
        list_evaluated_prob = []
        for c in torch.unique(self.z):
            extracted = self.target_samples[self.z == c]
            nu_n_c, lbda_n_c, mu_n_c, psi_n_c = self.compute_posterior_parameters(extracted)
            list_weight.append(extracted.shape[0] / (x.shape[0]))
            list_evaluated_prob.append(torch.exp(pyro.distributions.MultivariateStudentT(nu_n_c - self.d + 1, mu_n_c,
                                                                                         torch.cholesky(psi_n_c * (
                                                                                                     lbda_n_c + 1) / (
                                                                                                                    lbda_n_c * (
                                                                                                                        nu_n_c - self.d + 1)))).log_prob(
                x)).unsqueeze(-1))
            temp = torch.exp(pyro.distributions.MultivariateStudentT(nu_n_c - self.d + 1, mu_n_c, torch.cholesky(
                psi_n_c * (lbda_n_c + 1) / (lbda_n_c * (nu_n_c - self.d + 1)))).log_prob(x))
        probs = torch.sum(torch.tensor(list_weight).unsqueeze(0) * torch.cat(list_evaluated_prob, dim=1), dim=-1)
        return torch.log(probs)

    def plot_assignation(self):
        matplotlib.pyplot.scatter(self.target_samples[:, 0].numpy(), self.target_samples[:, 1].numpy(), c=self.z, cmap=matplotlib.cm.get_cmap('plasma'), alpha=.5)

    def train(self, epochs,verbose = False, visual = False):
        if visual:
            assert self.d == 2, 'visual not implemented for dimension different than 2'
            self.plot_assignation()
            plt.show(block=False)
            plt.pause(3)
            plt.close()
            plot_2d_function(lambda x: torch.exp(self.log_prob(x)), range=[[-3,3],[-3,3]], bins = (200,200))
            plt.show(block=False)
            plt.pause(3)
            plt.close()
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            for i in range(self.target_samples.shape[0]):
                probs = self.compute_probability(i)
                temp = torch.cat([self.z[:i], self.z[i + 1:]], dim=0)
                list_z_i = torch.cat([torch.unique(temp), (torch.max(torch.unique(temp)) + 1).unsqueeze(-1)])
                self.z[i] = list_z_i[torch.distributions.Categorical(probs / torch.sum(probs)).sample()]
            if visual:
                self.plot_assignation()
                plt.show(block=False)
                plt.pause(3)
                plt.close()
                plot_2d_function(lambda x: torch.exp(self.log_prob(x)), range=[[-3, 3], [-3, 3]], bins=(200, 200))
                plt.show(block=False)
                plt.pause(3)
                plt.close()

class MixtureDirichletProcess(torch.nn.Module):
    def __init__(self, target_samples, nu = torch.tensor(3.), lbda = torch.tensor(0.01) ,alpha = torch.tensor(50.), truncation = 20):
        super().__init__()
        self.target_samples = target_samples
        self.d = target_samples.shape[-1]
        self.nu = nu
        self.lbda = lbda
        self.mu = torch.mean(target_samples,dim =0)
        self.psi = torch.eye(self.d)
        self.alpha = alpha
        self.z = torch.zeros(target_samples.shape[0])
        self.truncation = truncation
        self.beta_distribution = torch.distributions.Beta(1, self.alpha)

        self.z = torch.randint(size=[self.target_samples.shape[0]], high=1)
        self.mean = torch.zeros(2).unsqueeze(0).repeat(1, 1)
        self.cov = torch.eye(2).unsqueeze(0).repeat(1, 1, 1)

    def sample_weights(self):
        counts = torch.unique(self.z, return_counts=True)[1]
        probs = torch.cat([counts, self.alpha.unsqueeze(-1)], dim=-1)
        w = torch.distributions.Dirichlet(probs).sample()
        r = w[-1]
        w_ = w[:-1]
        for i in range(self.truncation - 1):
            v = self.beta_distribution.sample()
            w = r * v
            r = r * (1 - v)
            w_ = torch.cat([w_, w.unsqueeze(-1)], dim=-1)
        w_ = torch.cat([w_, r.unsqueeze(-1)], dim=-1)
        return w_

    def sample_prior_parameters(self,num_samples):
        sigma = torch.inverse(torch.distributions.Wishart(self.nu, torch.inverse(self.psi)).sample(num_samples))
        sigma = (sigma + torch.transpose(sigma, 1, 2)) / 2
        mean = torch.distributions.MultivariateNormal(self.mu, scale_tril=torch.cholesky(sigma) / (self.lbda ** (1 / 2))).sample()
        return mean, sigma

    def repopulate_parameters(self):
        _mean, _cov = self.sample_prior_parameters([self.truncation])
        self.mean = torch.cat([self.mean, _mean], dim=0)
        self.cov = torch.cat([self.cov, _cov], dim=0)

    def plot_assignation(self):
        plt.scatter(self.target_samples[:, 0].numpy(), self.target_samples[:, 1].numpy(), c=self.z, cmap=matplotlib.cm.get_cmap('plasma'), alpha=.5)

    def log_prob(self,x):
        unique, count = torch.unique(self.z, return_counts=True)
        weights = count / self.z.shape[0]
        distribution = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(weights),
                                                             torch.distributions.MultivariateNormal(self.mean, self.cov))
        return distribution.log_prob(x)

    def equivalent_allocation(self):
        unique, self.z = torch.unique(self.z, return_inverse=True)
        self.mean = self.mean[unique]
        self.cov = self.cov[unique]

    def sample_parameter_posterior(self):
        list_mean = []
        list_cov = []
        for c in torch.unique(self.z):
            x_c = self.target_samples[self.z == c]
            N_c = x_c.shape[0]
            empirical_mean = torch.mean(x_c, dim=0)
            temp = (empirical_mean - self.mu).unsqueeze(-1)
            cov_c = torch.inverse(torch.distributions.Wishart(self.nu + N_c, torch.inverse(
                self.psi + (torch.cov(x_c.T) * (N_c - 1) if N_c >= 2 else torch.zeros(self.d)) + (self.lbda * N_c * temp @ temp.T) / (
                            self.lbda + N_c))).sample())
            cov_c = (cov_c + torch.transpose(cov_c, 0, 1)) / 2
            mean_c = torch.distributions.MultivariateNormal((self.lbda * self.mu + N_c * empirical_mean) / (self.lbda + N_c),
                                                            scale_tril=torch.cholesky(cov_c) / (self.lbda + N_c) ** (
                                                                        1 / 2)).sample()
            list_cov.append(cov_c.unsqueeze(0))
            list_mean.append(mean_c.unsqueeze(0))
        return torch.cat(list_mean, dim=0), torch.cat(list_cov, dim=0)

    def sample_allocation(self,w):
        w_i_k = w.unsqueeze(0).repeat(self.target_samples.shape[0], 1)
        p_i_k = torch.exp(
            torch.distributions.MultivariateNormal(self.mean, self.cov).log_prob(self.target_samples.unsqueeze(1).repeat(1, w.shape[0], 1)))
        temp = p_i_k * w_i_k
        self.z = torch.distributions.Categorical(temp / torch.sum(temp, dim=-1).unsqueeze(-1)).sample()

    def train(self, epochs, verbose = False, visual = False):
        if visual:
            self.plot_assignation()
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            plot_2d_function(lambda x: torch.exp(self.log_prob(x)), range=[[torch.min(self.target_samples[:,0]), torch.max(self.target_samples[:,0])], [torch.min(self.target_samples[:,1]), torch.max(self.target_samples[:,1])]], bins=(200, 200))
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            w = self.sample_weights()
            self.repopulate_parameters()
            self.sample_allocation(w)
            self.equivalent_allocation()
            if verbose:
                pbar.set_postfix_str('number of components = ' + str(self.mean.shape[0]) + '; log_prob = ' +str(torch.mean(self.log_prob(self.target_samples)).item()))
            self.sample_parameter_posterior()
            if visual:
                self.plot_assignation()
                plt.show(block=False)
                plt.pause(1)
                plt.close()
                plot_2d_function(lambda x: torch.exp(self.log_prob(x)),
                                 range=[[torch.min(self.target_samples[:, 0]), torch.max(self.target_samples[:, 0])],
                                        [torch.min(self.target_samples[:, 1]), torch.max(self.target_samples[:, 1])]],
                                 bins=(200, 200))
                plt.show(block=False)
                plt.pause(1)
                plt.close()

