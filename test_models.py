import torch
from classifiers import *
from utils.hpd import *

target_1_samples = torch.randn(1000,1)
target_0_samples = torch.randn(750,1)

plot_expected_coverage_1d_samples(target_1_samples,target_0_samples)
