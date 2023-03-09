import torch
from tqdm import tqdm

class regressor(torch.nn.Module):
    def __init__(self, D_input,D_output, hidden_dims):
        super().__init__()
        assert D_input.shape[0]==D_output.shape[0], 'Mismatch in number of samples'
        self.D_input = D_input
        self.D_output = D_output
        self.input_dim = D_input.shape[-1]
        self.output_dim = D_output.shape[-1]
        network_dimensions = [self.input_dim] + hidden_dims + [self.output_dim]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.SiLU(), ])
        network.pop()
        self.f = torch.nn.Sequential(*network)

        self.w = torch.distributions.Dirichlet(torch.ones(self.D_input.shape[0])).sample()

    def loss(self, input, output, w):
        return torch.sum(w * torch.norm(self.f(input)-output))

    def train(self, epochs, batch_size = None,verbose = False, lr = 5e-3, weight_decay=5e-6):
        if batch_size is None:
            batch_size = self.D_input.shape[0]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        if batch_size is None:
            batch_size = self.samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        dataset = torch.utils.data.TensorDataset(self.D_input, self.D_output, self.w)

        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for _ in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _,batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device), batch[2].to(device))
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0].to(device), batch[1].to(device), batch[2].to(device)) for _, batch in  enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss,4)) + '; device = ' + str(device))
        self.cpu()