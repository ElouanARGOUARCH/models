import torch
from density_estimation import *
from variational_inference import *
from utils.visual import *
from targets import *

target = TwoCircles()
target.visual()
mixt_dirichlet = MixtureDirichletProcess(target.sample([1000]))
mixt_dirichlet.train(100,verbose = True,visual = True)
