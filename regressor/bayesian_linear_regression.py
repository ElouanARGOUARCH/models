import torch


class generative_bayesian_linear_regression:
    #Models the data generative process
    def __init__(self, sigma2_simulateur, mu_X=torch.tensor(0.), sigma2_X=torch.tensor(1.),
                 mu_theta=torch.tensor(0.), sigma2_theta=torch.tensor(1.)):
        self.sigma2_simulateur = sigma2_simulateur
        self.mu_X = mu_X
        self.sigma2_X = sigma2_X
        self.mu_theta = mu_theta
        self.sigma2_theta = sigma2_theta

    def compute_theta_given_D_moments(self, DX, DY):
        assert DX.shape == DY.shape, 'Mismatch in number samples'
        sigma2_theta_given_D = 1/(DX @ DX / self.sigma2_simulateur + 1/self.sigma2_theta)
        mu_theta_given_D = sigma2_theta_given_D * (
                    DX @ DX / self.sigma2_simulateur + self.mu_theta/self.sigma2_theta)
        return mu_theta_given_D, sigma2_theta_given_D

    def compute_x0_given_y0_theta_moments(self, y0, theta):
        sigma2_x0_given_y0_theta = 1 / (
                    1 / self.sigma2_theta + y0.shape[0] * torch.square(theta) / self.sigma2_simulateur)
        mu_x0_given_y0_theta = sigma2_x0_given_y0_theta * (
                    self.mu_theta / self.sigma2_theta + theta * torch.sum(y0) / self.sigma2_simulateur)
        return mu_x0_given_y0_theta, sigma2_x0_given_y0_theta

    def log_joint_distribution(self, x0,theta,y0,DX, DY):
        mean_theta_given_D, var_theta_given_D = self.compute_theta_given_D_moments(DX, DY)
        theta_given_D_log_posterior = torch.distributions.Normal(mean_theta_given_D, torch.sqrt(var_theta_given_D)).log_prob(theta)

        x0_log_prior = torch.distributions.Normal(self.mu_X, torch.sqrt(self.sigma2_X)).log_prob(x0)

        mean = theta * y0
        var = self.sigma2_simulateur.repeat(y0.shape[0])
        y0_given_x0_theta_log_likelihood = torch.sum(torch.distributions.Normal(mean.unsqueeze(1).repeat(1, x0.shape[0]),
                                                              torch.sqrt(var).unsqueeze(1).repeat(1, x0.shape[
                                                                  0])).log_prob(x0.unsqueeze(0).repeat(mean.shape[0], 1)), dim=-1)

        return theta_given_D_log_posterior + x0_log_prior + y0_given_x0_theta_log_likelihood

    def marginal_log_posterior(self, x0, y0, D_theta, D_x):
        x0_log_prior = torch.distributions.Normal(self.mu_X, self.sigma2_X).log_prob(x0)
        mu_theta_given_D, sigma2_theta_given_D = self.compute_theta_given_D_moments(D_theta, D_x)
        gamma = y0.unsqueeze(-1).repeat(1, x0.shape[0])
        y0_given_x0_D_log_likelihood = torch.distributions.MultivariateNormal(mu_theta_given_D * gamma,
                                                            gamma.unsqueeze(-1) @ gamma.unsqueeze(
                                                                -2) * sigma2_theta_given_D + torch.eye(
                                                                y0.shape[0]) * self.sigma2_simulateur).log_prob(
            y0.unsqueeze(0).repeat(x0.shape[0], 1))
        return x0_log_prior + y0_given_x0_D_log_likelihood