import torch
from tqdm import tqdm

class discriminative_bayesian_linear_regression:
    def __init__(self, sigma2_simulateur, mu_X=torch.tensor(0.), sigma2_X=torch.tensor(1.),
                 mu_theta=torch.tensor(0.), sigma2_theta=torch.tensor(1.)):
        self.sigma2_simulateur = sigma2_simulateur
        self.mu_X = mu_X
        self.sigma2_X = sigma2_X
        self.mu_theta = mu_theta
        self.sigma2_theta = sigma2_theta

    def compute_theta_given_D_moments(self, DX, DY):
        assert DX.shape == DY.shape, 'Mismatch in number samples'
        sigma2_theta_given_D = 1/(DY @ DY / self.sigma2_simulateur + 1/self.sigma2_theta)
        mu_theta_given_D = sigma2_theta_given_D * (
                    DX @ DY / self.sigma2_simulateur + self.mu_theta/self.sigma2_theta)
        print(mu_theta_given_D)
        return mu_theta_given_D, sigma2_theta_given_D

    def compute_x0_given_y0_theta_moments(self, y0, theta):
        assert y0.shape[0] == 1, 'Discriminative does not support multiple observations'
        mu_x0_given_y0_theta = theta*y0
        sigma2_x0_given_y0_theta = self.sigma2_simulateur
        return mu_x0_given_y0_theta, sigma2_x0_given_y0_theta

    def compute_x0_given_y0_D_moments(self,y0, DX,DY):
        assert y0.shape[0]==1, 'Discriminative does not support multiple observations'
        mu_theta_given_D, sigma2_theta_given_D = self.compute_theta_given_D_moments(DX, DY)
        mu_x0_given_y0_D = mu_theta_given_D*y0
        sigma2_x0_given_y0_D = torch.square(y0)*sigma2_theta_given_D + self.sigma2_simulateur
        return mu_x0_given_y0_D, sigma2_x0_given_y0_D

class generative_bayesian_linear_regression:
    #Models the data generative process as Normal distribution with mean linear function and var fixed and known sigma2 and supposes a Normal prior on X
    def __init__(self, sigma2_simulateur, mu_X=torch.tensor(0.), sigma2_X=torch.tensor(1.),
                 mu_theta=torch.tensor(0.), sigma2_theta=torch.tensor(1.)):
        self.sigma2_simulateur = sigma2_simulateur
        self.mu_X = mu_X
        self.sigma2_X = sigma2_X
        self.mu_theta = mu_theta
        self.sigma2_theta = sigma2_theta

    def compute_theta_given_D_moments(self, DX, DY):
        #model posterior
        assert DX.shape == DY.shape, 'Mismatch in number samples'
        sigma2_theta_given_D = 1/(DX @ DX / self.sigma2_simulateur + 1/self.sigma2_theta)
        mu_theta_given_D = sigma2_theta_given_D * (
                    DY @ DX / self.sigma2_simulateur + self.mu_theta/self.sigma2_theta)
        return mu_theta_given_D, sigma2_theta_given_D

    def compute_x0_given_y0_theta_moments(self, y0, theta):
        #posterior
        sigma2_x0_given_y0_theta = 1 / (
                    1 / self.sigma2_theta + y0.shape[0] * torch.square(theta) / self.sigma2_simulateur)
        mu_x0_given_y0_theta = sigma2_x0_given_y0_theta * (
                    self.mu_theta / self.sigma2_theta + theta * torch.sum(y0) / self.sigma2_simulateur)
        return mu_x0_given_y0_theta, sigma2_x0_given_y0_theta

    def x0_theta_given_y0_D_log_joint(self, x0,theta,y0,DX, DY):
        #Joint posterior
        assert x0.shape[0]==theta.shape[0],'Mismatch in number of samples'
        mean_theta_given_D, var_theta_given_D = self.compute_theta_given_D_moments(DX, DY)
        theta_given_D_log_posterior = torch.distributions.Normal(mean_theta_given_D, torch.sqrt(var_theta_given_D)).log_prob(theta)

        x0_log_prior = torch.distributions.Normal(self.mu_X, torch.sqrt(self.sigma2_X)).log_prob(x0)

        mean = theta * x0
        var = self.sigma2_simulateur.repeat(mean.shape[0])
        y0_given_x0_theta_log_likelihood = torch.sum(torch.distributions.Normal(mean.unsqueeze(1).repeat(1, y0.shape[0]),
                                                              torch.sqrt(var).unsqueeze(1).repeat(1, y0.shape[
                                                                  0])).log_prob(y0.unsqueeze(0).repeat(mean.shape[0], 1)), dim=-1)

        return theta_given_D_log_posterior + x0_log_prior + y0_given_x0_theta_log_likelihood

    def x0_given_y0_D_marginal_log_posterior(self, x0, y0, DX, DY):
        #posterior predictive
        x0_log_prior = torch.distributions.Normal(self.mu_X, self.sigma2_X).log_prob(x0)
        mu_theta_given_D, sigma2_theta_given_D = self.compute_theta_given_D_moments(DX, DY)
        gamma = x0.unsqueeze(-1).repeat(1, y0.shape[0])
        y0_given_x0_D_marginal_log_likelihood = torch.distributions.MultivariateNormal(mu_theta_given_D * gamma,
                                                            gamma.unsqueeze(-1) @ gamma.unsqueeze(-2) * sigma2_theta_given_D + torch.eye(y0.shape[0]) * self.sigma2_simulateur).log_prob(y0.unsqueeze(0).repeat(x0.shape[0], 1))
        return x0_log_prior + y0_given_x0_D_marginal_log_likelihood

    def sample_x0_given_y0_D_gibbs(self, y0,DX, DY,number_steps,verbose = False):
        #sample
        DYplus = torch.cat([DY, y0], dim=0)
        mu_theta_given_D, sigma2_theta_given_D = self.compute_theta_given_D_moments(DX, DY)
        current_theta = torch.distributions.Normal(mu_theta_given_D, torch.sqrt(sigma2_theta_given_D)).sample()

        mu_x0_given_y0_theta, sigma2_x0_given_y0_theta = self.compute_x0_given_y0_theta_moments(y0, current_theta)
        current_x0 = torch.distributions.Normal(mu_x0_given_y0_theta, torch.sqrt(sigma2_x0_given_y0_theta)).sample()
        list_x0_gibbs = []
        list_theta_gibbs = []
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0])], dim=0)
            mu_theta_given_Dplus, sigma2_theta_given_Dplus = self.compute_theta_given_D_moments(DXplus, DYplus)
            current_theta = torch.distributions.Normal(mu_theta_given_Dplus, torch.sqrt(sigma2_theta_given_Dplus)).sample()
            mu_x0_given_y0_theta, sigma2_x0_given_y0_theta = self.compute_x0_given_y0_theta_moments(y0, current_theta)
            current_x0 = torch.distributions.Normal(mu_x0_given_y0_theta, torch.sqrt(sigma2_x0_given_y0_theta)).sample()
            list_x0_gibbs.append(current_x0)
            list_theta_gibbs.append(current_theta)
        return torch.stack(list_x0_gibbs), torch.stack(list_theta_gibbs)

class generative_bayesian_affine_regression:
    def __init__(self, sigma2_simulateur, mu_X=torch.tensor(0.), sigma2_X=torch.tensor(1.),
                 mu_theta=torch.zeros(2), Sigma_theta=torch.eye(2)):
        self.sigma2_simulateur = sigma2_simulateur
        self.mu_X = mu_X
        self.sigma2_X = sigma2_X
        self.mu_theta = mu_theta
        self.Sigma_theta = Sigma_theta

    def compute_x0_given_y0_theta_moments(self, y0, theta):
        sigma_x0_given_y0_theta = 1 / (
                    1 / self.sigma2_X + (y0.shape[0] * theta[0] ** 2) / self.sigma2_simulateur)
        mu_x0_given_y0_theta = sigma_x0_given_y0_theta * (
                    self.mu_X / self.sigma2_X + theta[0] * torch.sum(y0 - theta[1]) / self.sigma2_simulateur)
        return mu_x0_given_y0_theta, sigma_x0_given_y0_theta

    def compute_theta_given_D_moments(self, DX, DY):
        assert DX.shape[0] == DY.shape[0], 'Mismatch in number samples'
        if DX.shape[0] >= 1:
            temp = torch.cat([DX.unsqueeze(-1), torch.ones(DX.shape[0], 1)], dim=-1)
            Sigma_theta_given_D = torch.inverse(
                temp.T @ temp / self.sigma2_simulateur + torch.inverse(self.Sigma_theta))
            mu_theta_given_D = Sigma_theta_given_D @ (
                        DY @ temp / self.sigma2_simulateur + torch.inverse(self.Sigma_theta)@self.mu_theta)
        else:
            mu_theta_given_D, Sigma_phi_given_D = self.mu_phi, self.sigma_phi
        return mu_theta_given_D, Sigma_theta_given_D

    def x0_theta_given_y0_D_log_joint(self, x0,theta,y0,DX,DY):
        assert x0.shape[0] == theta.shape[0],'Mismatch in number of samples'
        x0_log_prior = torch.distributions.Normal(self.mu_X, torch.sqrt(self.sigma2_X)).log_prob(x0)

        mean_theta_given_D, Sigma_theta_given_D = self.compute_theta_given_D_moments(DX, DY)
        theta_given_D_log_posterior = torch.distributions.MultivariateNormal(mean_theta_given_D,
                                                                 Sigma_theta_given_D).log_prob(theta)

        augmented_x0 = torch.cat([x0.unsqueeze(-1), torch.ones(x0.shape[0], 1)], dim=-1)
        temp = torch.bmm(theta.unsqueeze(-2), augmented_x0.unsqueeze(-1)).squeeze(-1)
        temp = temp.repeat(1, y0.shape[0])
        cov_matrix = self.sigma2_simulateur * torch.eye(y0.shape[0]).unsqueeze(0).repeat(theta.shape[0], 1, 1)
        y0_given_x0_theta_log_likelihood = torch.distributions.MultivariateNormal(temp, cov_matrix).log_prob(y0) if \
        y0.shape[0] >= 1 else torch.zeros(theta.shape[0])

        return theta_given_D_log_posterior + x0_log_prior + y0_given_x0_theta_log_likelihood

    def x0_given_y0_D_marginal_log_posterior(self, x0, y0, DX, DY):
        mean_theta_given_D, Sigma_theta_given_D = self.compute_theta_given_D_moments(DX, DY)
        gamma = torch.cat([x0.unsqueeze(-1), torch.ones_like(x0).unsqueeze(-1)], dim=-1).unsqueeze(-2).repeat(
            1, y0.shape[0], 1)
        x0_log_prior = torch.distributions.Normal(self.mu_X, self.sigma2_X).log_prob(x0)
        yO_given_x0_D_marginal_log_likelihood = torch.distributions.MultivariateNormal(gamma @ mean_theta_given_D,
                                                            gamma @ Sigma_theta_given_D @ gamma.mT + self.sigma2_simulateur * torch.eye(
                                                                x0.shape[0])).log_prob(
            x0.unsqueeze(0).repeat(x0.shape[0], 1))
        return x0_log_prior + yO_given_x0_D_marginal_log_likelihood

    def sample_x0_given_y0_D_gibbs(self, y0,DX, DY,number_steps,verbose = False):
        DYplus = torch.cat([DY, y0], dim=0)
        mean_theta_given_D, Sigma_theta_given_D = self.compute_theta_given_D_moments(DX, DY)
        current_theta = torch.distributions.MultivariateNormal(mean_theta_given_D, Sigma_theta_given_D).sample()

        mean_x0_given_y0_theta, sigma2_x0_given_y0_theta = self.compute_x0_given_y0_theta_moments(y0, current_theta)
        current_x0 = torch.distributions.Normal(mean_x0_given_y0_theta, torch.sqrt(sigma2_x0_given_y0_theta)).sample()
        list_x0_gibbs = []
        list_theta_gibbs = []
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0])], dim=0)
            mean_theta_given_Dplus, Sigma_theta_given_Dplus = self.compute_theta_given_D_moments(DXplus, DYplus)
            current_theta = torch.distributions.MultivariateNormal(mean_theta_given_Dplus, Sigma_theta_given_Dplus).sample()
            mean_x0_given_y0_theta, sigma2_x0_given_y0_theta = self.compute_x0_given_y0_theta_moments(y0, current_theta)
            current_x0 = torch.distributions.Normal(mean_x0_given_y0_theta, torch.sqrt(sigma2_x0_given_y0_theta)).sample()
            list_x0_gibbs.append(current_x0)
            list_theta_gibbs.append(current_theta)
        return torch.stack(list_x0_gibbs), torch.stack(list_theta_gibbs)