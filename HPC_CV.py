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

Sb0 = 		data['Sb0']
Sqs = 		data['Sqs']
qs = 		data['qs']
thetas = 	data['thetas']
weights = 	data['weights']
mask = 		data['mask']
lin2idx = 	data['lin2idx']
std_vector= data['std_vector']

###########################################
# LOOCV
###########################################

R24 = rrt.R_mask(mask, 24, order=2)

alpha_list = torch.tensor([0.4, 0.6, 0.8, 1.0, 1.2, 1.4]) # 
beta_list = torch.tensor([0, 1e0, 1e1, 1e2, 1e3, 1e4]) # 

combines, _ = bft.Cartesian(alpha_list, beta_list)

loss_list = torch.zeros(combines.shape[0])
history_list = [None] * (combines.shape[0])

# -----------------------------------------
start = 0
# -----------------------------------------

for k in range(start, combines.shape[0]):

	print("alpha=", combines[k, 0], "beta=", combines[k, 1])

	loss = 0
	for i in range(qs.shape[0]):

		t0 = time.perf_counter()
		_, f_CV, history_CV = dast.admm(qs[torch.arange(qs.shape[0]) != i], 
										thetas, 
										weights, 
										Sqs[mask, :][:, torch.arange(Sqs[mask, :].shape[1]) != i],
										sigma = combines[k, 0] * std_vector.repeat_interleave(24),
										R_csr = combines[k, 1] * R24, 
										f0 = None, normalize = True, Lambdas = None, 
										beta = 0.5, c = 1e-4, epsilon = 1e-8, tol = 1e-12, maxiter = 1,
										cg_rtol = 1e-10, cg_atol = 1e-50, cg_maxiter = 1000, 
										rho = 0.5, rho_ratio = 2, dynamic_rho = True,
										admm_tol = 1e-10, admm_maxiter = 200)
		
		t_CV = time.perf_counter() - t0
		print(f"{i}th, time: {t_CV:.3f} s")

		Sqs_CV = bft.get_Sqs(bft.kernel(qs, thetas), weights, f_CV)
		history_list[k].append(history_CV)				
		loss += torch.sum((Sqs_CV[:, i] - Sqs[mask, :][:, i])**2)

		del f_CV, Sqs_CV, history_CV
		gc.collect()
		torch.cuda.empty_cache()

	loss_list[k] = loss
	print("+-----------------------------------------------------------------------------+")
		
	state = {
		"combines": combines,
		"loss_list": loss_list,
		"history_list": history_list}
		
	torch.save(state, f"/work/xunan/MaxEnt/Results/CV_results_from{start}.pt")

