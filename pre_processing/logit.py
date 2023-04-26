import torch

class logit():
    def __init__(self, alpha):
        self.alpha = alpha

    def transform(self,x, alpha = None):
        if alpha is None:
            alpha = self.alpha
        return torch.logit(alpha*torch.ones_like(x) + x*(1-2*alpha))

    def inverse_transform(self, x, alpha = None):
        if alpha is None:
            alpha = self.alpha
        return torch.sigmoid((x- alpha*torch.ones_like(x))/(1-2*alpha))