import torch
import pyro
import matplotlib
from utils.visual import *
from tqdm import tqdm

class MixtureDirichletProcess(torch.nn.Module):
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

    def plot_assignation(self,fig):
        fig.scatter(self.target_samples[:, 0].numpy(), self.target_samples[:, 1].numpy(), c=self.z, cmap=matplotlib.cm.get_cmap('plasma'), alpha=.5)

    def train(self, epochs,verbose = False, visual = False):
        if visual:
            assert self.d == 2, 'visual not implemented for dimension different than 2'
            fig = matplotlib.pyplot.figure()
            ax = fig.add_subplot(111)
            self.plot_assignation(ax)
            matplotlib.pyplot.show(block=False)
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
                matplotlib.pyplot.close(fig)
                fig = matplotlib.pyplot.figure()
                ax = fig.add_subplot(111)
                self.plot_assignation(ax)
                matplotlib.pyplot.show(block=False)


