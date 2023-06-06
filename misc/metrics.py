import torch

def compute_accuracy(log_prob, labels):
    assert log_prob.shape[0] == labels.shape[0], "wrong number of samples"
    if labels.shape == log_prob.shape[:-1]:
        labels = torch.nn.functional.one_hot(labels, num_classes = log_prob.shape[-1])
    temp = torch.abs(torch.nn.functional.one_hot(torch.argmax(log_prob,dim = -1), num_classes =log_prob.shape[-1]) - labels)
    return (log_prob.shape[0] - torch.sum(temp)/2)/log_prob.shape[0]