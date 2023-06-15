import torch
import pyro
from tqdm import tqdm

class discriminative_bayesian_linear_regression_known_variance:
    def __init__(self, sigma2_simulateur,mu_beta=torch.tensor(0.), sigma2_beta=torch.tensor(1.)):
        self.sigma2_simulateur = sigma2_simulateur
        self.mu_beta = mu_beta
        self.sigma2_beta = sigma2_beta

    def compute_beta_given_D_moments(self, DX, DY):
        assert DX.shape == DY.shape, 'Mismatch in number samples'
        sigma2_beta_given_D = 1/(DY @ DY / self.sigma2_simulateur + 1/self.sigma2_beta)
        mu_beta_given_D = sigma2_beta_given_D * (
                    DX @ DY / self.sigma2_simulateur + self.mu_beta/self.sigma2_beta)
        return mu_beta_given_D, sigma2_beta_given_D

    def compute_x0_given_y0_beta_moments(self, y0, beta):
        assert y0.shape[0] == 1, 'Discriminative does not support multiple observations'
        mu_x0_given_y0_beta = beta*y0
        sigma2_x0_given_y0_beta = self.sigma2_simulateur
        return mu_x0_given_y0_beta, sigma2_x0_given_y0_beta

    def compute_x0_given_y0_D_moments(self,y0, DX,DY):
        assert y0.shape[0]==1, 'Discriminative does not support multiple observations'
        mu_beta_given_D, sigma2_beta_given_D = self.compute_beta_given_D_moments(DX, DY)
        mu_x0_given_y0_D = mu_beta_given_D*y0
        sigma2_x0_given_y0_D = torch.square(y0)*sigma2_beta_given_D + self.sigma2_simulateur
        return mu_x0_given_y0_D, sigma2_x0_given_y0_D

class discriminative_bayesian_affine_regression_known_variance:
    def __init__(self, sigma2_simulateur,mu_beta=torch.zeros(2), Sigma_beta=torch.eye(2)):
        self.sigma2_simulateur = sigma2_simulateur
        self.mu_beta = mu_beta
        self.Sigma_beta = Sigma_beta

    def compute_beta_given_D_moments(self, DX, DY):
        assert DX.shape == DY.shape, 'Mismatch in number samples'
        temp = torch.cat([DY.unsqueeze(-1), torch.ones(DY.shape[0], 1)], dim=-1)
        Sigma_beta_given_D = torch.inverse(temp.T @ temp / self.sigma2_simulateur + torch.inverse(self.Sigma_beta))
        mu_beta_given_D = Sigma_beta_given_D@(DX @ temp / self.sigma2_simulateur + torch.inverse(self.Sigma_beta)@self.mu_beta)
        return mu_beta_given_D, Sigma_beta_given_D

    def compute_x0_given_y0_beta_moments(self, y0, beta):
        assert y0.shape[0] == 1, 'Discriminative does not support multiple observations'
        mu_x0_given_y0_beta = beta@torch.cat([y0,torch.ones_like(y0)], dim = -1)
        sigma2_x0_given_y0_beta = self.sigma2_simulateur
        return mu_x0_given_y0_beta, sigma2_x0_given_y0_beta

    def compute_x0_given_y0_D_moments(self,y0, DX,DY):
        assert y0.shape[0]==1, 'Discriminative does not support multiple observations'
        mu_beta_given_D, sigma2_beta_given_D = self.compute_beta_given_D_moments(DX, DY)
        mu_x0_given_y0_D = mu_beta_given_D@torch.cat([y0,torch.ones_like(y0)], dim = -1)
        sigma2_x0_given_y0_D = torch.cat([y0,torch.ones_like(y0)], dim = -1)@sigma2_beta_given_D@torch.cat([y0,torch.ones_like(y0)], dim = -1) + self.sigma2_simulateur
        return mu_x0_given_y0_D, sigma2_x0_given_y0_D.squeeze(-1)

class discriminative_bayesian_affine_regression:
    def __init__(self,mu_beta=torch.zeros(2), Sigma_beta=torch.eye(2), shape_sigma2=torch.tensor(1.),
                 scale_sigma2=torch.tensor(1.)):
        self.mu_beta = mu_beta
        self.Sigma_beta = Sigma_beta
        self.shape_sigma2 = shape_sigma2
        self.scale_sigma2 = scale_sigma2

    def compute_beta_given_sigma2_D_moments(self,sigma2, DX, DY):
        assert DX.shape == DY.shape, 'Mismatch in number samples'
        temp = torch.cat([DY.unsqueeze(-1), torch.ones(DY.shape[0], 1)], dim=-1)
        Sigma_beta_given_D = torch.inverse(temp.T @ temp / sigma2 + torch.inverse(self.Sigma_beta))
        mu_beta_given_D = Sigma_beta_given_D@(DX @ temp / sigma2 + torch.inverse(self.Sigma_beta)@self.mu_beta)
        return mu_beta_given_D, Sigma_beta_given_D

    def compute_x0_given_y0_beta_sigma2_moments(self, y0, beta,sigma2):
        assert y0.shape[0] == 1, 'Discriminative does not support multiple observations'
        mu_x0_given_y0_beta = beta@torch.cat([y0,torch.ones_like(y0)], dim = -1)
        sigma2_x0_given_y0_beta = sigma2
        return mu_x0_given_y0_beta, sigma2_x0_given_y0_beta

    def compute_x0_given_y0_D_moments(self,y0, DX,DY):
        assert y0.shape[0]==1, 'Discriminative does not support multiple observations'
        mu_beta_given_D, sigma2_beta_given_D = self.compute_beta_given_D_moments(DX, DY)
        mu_x0_given_y0_D = mu_beta_given_D@torch.cat([y0,torch.ones_like(y0)], dim = -1)
        sigma2_x0_given_y0_D = torch.cat([y0,torch.ones_like(y0)], dim = -1)@sigma2_beta_given_D@torch.cat([y0,torch.ones_like(y0)], dim = -1) + self.sigma2_simulateur
        return mu_x0_given_y0_D, sigma2_x0_given_y0_D.squeeze(-1)

    def compute_sigma2_given_beta_D_parameters(self, beta, DX, DY):
        assert DX.shape[0] == DY.shape[0], 'Mismatch in number samples'
        temp = torch.cat([DY.unsqueeze(-1), torch.ones(DY.shape[0], 1)], dim=-1)
        shape_N = self.shape_sigma2 + DX.shape[0] / 2
        scale_N = self.scale_sigma2 + torch.sum(torch.square(DX - temp @ beta)) / 2
        estimated_sigma2 = pyro.distributions.InverseGamma(shape_N, scale_N).sample()
        return estimated_sigma2

    def sample_x0_given_y0_D_Y_gibbs(self, y0, DX, DY,number_steps=100, verbose=False):
        assert DX.shape[0] == DY.shape[0], 'mismatch in dataset numbers'
        current_sigma2 = pyro.distributions.InverseGamma(self.shape_sigma2, self.scale_sigma2).sample()
        mean_beta_given_sigma2_D, Sigma_beta_given_sigma2_D = self.compute_beta_given_sigma2_D_moments(current_sigma2, DX, DY)
        current_beta = torch.distributions.MultivariateNormal(mean_beta_given_sigma2_D, Sigma_beta_given_sigma2_D).sample()
        list_x0_gibbs = []
        list_beta_gibbs = []
        list_sigma2_gibbs = []
        DYplus = torch.cat([DY, y0], dim = 0)
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            mean_x0_given_y0_beta_sigma2, sigma2_x0_given_y0_beta_sigma2 = self.compute_x0_given_y0_beta_sigma2_moments(
                y0, current_beta, current_sigma2)
            current_x0 = torch.distributions.Normal(mean_x0_given_y0_beta_sigma2,
                                                    torch.sqrt(sigma2_x0_given_y0_beta_sigma2)).sample()
            DXplus = torch.cat([DX, current_x0.reshape(y0.shape[0])])
            mean_beta_given_Dplus, Sigma_beta_given_Dplus = self.compute_beta_given_sigma2_D_moments(current_sigma2,
                                                                                                     DXplus, DYplus)
            current_beta = torch.distributions.MultivariateNormal(mean_beta_given_Dplus,
                                                                  Sigma_beta_given_Dplus).sample()
            current_sigma2 = self.compute_sigma2_given_beta_D_parameters(current_beta, DXplus, DYplus)
            list_x0_gibbs.append(current_x0)
            list_beta_gibbs.append(current_beta)
            list_sigma2_gibbs.append(current_sigma2)
        return torch.stack(list_x0_gibbs), torch.stack(list_beta_gibbs), torch.stack(list_sigma2_gibbs)