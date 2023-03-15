import torch
from classifiers import *

target_1_samples = torch.randn(1000,1)
target_0_samples = torch.randn(750,1)
binary_classifier = BinaryClassifier(target_0_samples, target_1_samples,[32,32])
binary_classifier.train(100)