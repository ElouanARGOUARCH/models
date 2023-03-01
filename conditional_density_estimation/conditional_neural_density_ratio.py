import torch
from tqdm import tqdm

class NeuralLikelihoodRatio(torch.nn.Module):
    def __init__(self, D_x, D_theta, hidden_dims):
        super().__init__()

        self.p = D_x.shape[-1]
        self.d = D_theta.shape[-1]
        assert D_x.shape[0] == D_theta.shape[0], "Number of samples do not match"
        self.D_x = D_x
        self.D_theta = D_theta

        self.w = torch.distributions.Dirichlet(torch.ones(self.D_x.shape[0])).sample()

        network_dimensions = [self.p + self.d] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.SiLU(), ])
        network.pop()
        self.logit_r = torch.nn.Sequential(*network)

        self.loss_values = []

    def loss(self, x, theta,w):
        log_sigmoid = torch.nn.LogSigmoid()
        x_tilde = x[torch.randperm(x.shape[0])]
        true = torch.cat([x, theta], dim=-1)
        fake = torch.cat([x_tilde, theta], dim=-1)
        return -torch.sum(w*(log_sigmoid(self.logit_r(true)) + log_sigmoid(-self.logit_r(fake))))

    def log_ratio(self,x,theta):
        return self.logit_r(torch.cat([x, theta], dim=-1)).squeeze(-1)

    def train(self, epochs, batch_size=None, lr = 5e-3, verbose = False):
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr)
        if batch_size is None:
            batch_size = self.D_x.shape[0]
        dataset = torch.utils.data.TensorDataset(self.D_x, self.D_theta, self.w)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                batch_loss = self.loss(batch[0].to(device), batch[1].to(device), batch[2].to(device))
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0].to(device),batch[1].to(device)) for i, batch in enumerate(dataloader)]).sum().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + '; device =' +str(device))
        self.to(torch.device('cpu'))