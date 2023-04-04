import torch
import matplotlib.pyplot as plt
from regressor import *
from utils import *

mu_X =torch.tensor(6.)
sigma2_X =torch.tensor(1.)
prior = torch.distributions.Normal(mu_X, torch.sqrt(sigma2_X))
n_D = 5
DX =torch.linspace(-1,1, n_D)
sigma2_simulateur = torch.tensor([0.25])
f = lambda y: 1*y
simulateur= lambda x: f(x) + torch.randn(x.shape[0])*torch.sqrt(sigma2_simulateur)

DY = simulateur(DX)

x0 = prior.sample()
x0 = torch.tensor([6.])
print('x0 = ',str(x0.item()))
n_y0= 1
y0 = simulateur(x0.repeat(n_y0))

plt.figure(figsize = (15,8))
y_min = torch.min(torch.cat([DY,y0])) - 0.5
plt.scatter(DX.numpy(), DY.numpy(), alpha =.5, label = 'D={(X,Y)}')

tt = torch.linspace(-5,10,5000)
plt.plot(tt, y_min + torch.exp(prior.log_prob(tt.unsqueeze(-1))), color = 'green', label='Prior')
plt.plot(tt.numpy(),f(tt).numpy(), linestyle = '--', label = 'Unknown Linear model')
plt.axvline(x0.numpy(), color = 'green', alpha = .7, linestyle = '--', label=' Unknown value x0')
hist = plt.hist(y0.numpy(), orientation ='horizontal', bins = 20, density = True, bottom = -5, label = 'histogram of y0')
plt.ylim(y_min,)
plt.legend()
plt.show()


blr = generative_bayesian_linear_regression(sigma2_simulateur, mu_X, sigma2_X, torch.tensor(0.), torch.tensor(1.))
plot_1d_unormalized_values(torch.exp(blr.x0_given_y0_D_marginal_log_posterior(tt, y0, DX, DY)),tt, show = False)

blr = discriminative_bayesian_linear_regression(sigma2_simulateur, mu_X, sigma2_X, torch.tensor(0.), torch.tensor(1.))
mean, sigma2= blr.compute_x0_given_y0_D_moments(y0, DX, DY)
posterior_predictive=torch.distributions.Normal(mean, torch.sqrt(sigma2))
plt.plot(tt, torch.exp(posterior_predictive.log_prob(tt)), label = 'Discriminative')
plt.legend()
plt.show()
