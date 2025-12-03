import torch
import time
import numpy as np

import sys
import importlib

import gc

sys.path.append("/work/xunan/MaxEnt/Code")

import Basic_functions_torch as bft
import R_roughness_torch as rrt
import D2_admm_solver_torch as dast

importlib.reload(bft)
importlib.reload(rrt)
importlib.reload(dast)

###########################################
# Set the global device and data type
###########################################

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)

###########################################
# Read the data
###########################################

data = torch.load('/work/xunan/MaxEnt/Data/Sigma_estimated.pt', map_location="cuda")

Sqs = 		data['Sqs']
qs = 		data['qs']
thetas = 	data['thetas']
weights = 	data['weights']
mask = 		data['mask']
lin2idx = 	data['lin2idx']
sigma = 	data['sigma']

###########################################
# Apply admm
###########################################

R0 = torch.sparse_csr_tensor(
    torch.zeros(torch.sum(mask)*25 + 1),
    torch.empty(0),
    torch.empty(0),
    size=(torch.sum(mask)*25, torch.sum(mask)*25),
    device='cuda'
)

R25 = rrt.R_mask(mask, 25, order=2)

t0 = time.perf_counter()

lambdas, f_hat, history = dast.admm(   	qs, thetas, weights, Sqs[mask, :], sigma = 0.5*sigma,
										R_csr = R0, f0 = None, normalize = True, Lambdas = None, 
										beta = 0.5, c = 1e-4, epsilon = 1e-8, tol = 1e-12, maxiter = 1,
										cg_rtol = 1e-10, cg_atol = 1e-50, cg_maxiter = 1000,
                                        rho = 0.5, rho_ratio = 2, dynamic_rho = True,
										admm_tol = 1e-10, admm_maxiter = 200)
t_admm = time.perf_counter() - t0

print("+-----------------------------------------------------------------------------+")
print(f"ADMM with R==0: {t_admm:.3f} s")
print("+-----------------------------------------------------------------------------+")

###########################################
# Output the results
###########################################

state = {
	"mask": mask,
	"lin2idx": lin2idx,
	"lambdas": lambdas,
	"history": history		}
	
torch.save(state, "/work/xunan/MaxEnt/Results/R_solution.pt")
del state

# del theta1, theta2, mask, lin2idx, lambdas, f_hat, history
# gc.collect()
# torch.cuda.empty_cache()


