import torch
from density_estimation import *
from variational_inference import *
from utils.visual import *
from targets import *

target = Orbits()
mixt_dirichlet = MixtureDirichletProcess(target.sample([5000]))
mixt_dirichlet.train(100,verbose = True,visual = True)
