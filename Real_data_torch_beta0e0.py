import torch
import time
import numpy as np
import nibabel as nib

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

img  = nib.load("/work/xunan/MaxEnt/Data/MultiTE_DirAvg/REDIM_diravg.nii.gz")

signal = torch.from_numpy(img.get_fdata()).flip(0).to('cuda') # data from 'from_numpy' will always be on the cpu

mask, lin2idx = bft.mask_brain(signal, median_radius = 1, numpass = 4, least_size = 100, keep_top = None)
		
bvals = torch.from_numpy(np.loadtxt("/work/xunan/MaxEnt/Data/MultiTE_DirAvg/bvals.txt")).to('cuda')/ 1000  # in unit ms/μm^2
TEs   = torch.from_numpy(np.loadtxt("/work/xunan/MaxEnt/Data/MultiTE_DirAvg/TEs.txt")).to('cuda')  / 1000  # in unit s

qs = torch.column_stack((bvals, TEs))  # shape (N, 2)
		
###########################################
# Process the data
###########################################
		
signal_norm = signal[:,:,:,qs[:,0]!=0] / signal[:,:,:,qs[:,0]==0].repeat_interleave(5, dim=-1)
qs_norm = qs[qs[:, 0] != 0]

Sqs = signal_norm[mask, :].reshape(-1, qs_norm.shape[0])

theta1 = torch.linspace(0, 3, 50)
theta2 = torch.linspace(0, 40, 50)

thetas, weights = bft.Cartesian(theta1, theta2)

R25 = rrt.R_mask(mask, 25, order=2)
R24 = rrt.R_mask(mask, 24, order=2)

##-----------------------------------------
del img, signal, bvals, TEs, qs, signal_norm
gc.collect()
torch.cuda.empty_cache()
##-----------------------------------------

###########################################
# Set R to zero to estimate noise level
###########################################

print("+-----------------------------------------------------------------------------+")

std_vector = 0.02 * torch.ones(len(Sqs))
sigma = std_vector.repeat_interleave(25)

for i in range(1):
	
	t0 = time.perf_counter()

	_, f_est, _ = dast.admm(	qs_norm, thetas, weights, Sqs, sigma = sigma,
								R_csr = 0 * R25, f0 = None, normalize = True,
								Lambdas = None, rho = 0.5, rho_ratio = 2, dynamic_rho = True,
								beta = 0.5, c = 1e-4, tol = 1e-6, epsilon = 1e-8, maxiter = 10,
								cg_rtol = 1e-10, cg_atol = 1e-50, cg_maxiter = 1000,
								admm_tol = 1e-10, admm_maxiter = 200)
	
	t_est = time.perf_counter() - t0
	print(f"Pre-estimation: {t_est:.3f} s")

	Sqs_est = bft.get_Sqs(bft.kernel(qs_norm, thetas), weights, f_est)
	diff = Sqs - Sqs_est
	std_vector = diff.std(dim=1)
	sigma = std_vector.repeat_interleave(25)

print("+-----------------------------------------------------------------------------+")
print("estimation of sigma is:", torch.mean(sigma))

##-----------------------------------------
del f_est, Sqs_est, diff, theta1, theta2, mask, lin2idx
gc.collect()
torch.cuda.empty_cache()
##-----------------------------------------

###########################################
# LOOCV
###########################################

alpha_list = torch.tensor([0.25, 0.5, 1, 1.5, 2]) 
beta_list = torch.tensor([0e0])

loss_table = torch.zeros((len(alpha_list), len(beta_list)))
history_list = [None] * (len(alpha_list) * len(beta_list))

for m, alpha in enumerate(alpha_list):
	for n, beta in enumerate(beta_list):

		print("Started: m=", m, "n=", n)

		loss = 0
		for i in range(qs_norm.shape[0]):

			t0 = time.perf_counter()
			_, f_CV, history_CV = dast.admm(qs_norm[torch.arange(qs_norm.shape[0]) != i], 
											thetas, 
											weights, 
											Sqs[:, torch.arange(Sqs.shape[1]) != i],
											sigma = alpha * std_vector.repeat_interleave(24),
											R_csr = beta * R24, 
											f0 = None, normalize = True,
											Lambdas = None, rho = 0.5, rho_ratio = 2, dynamic_rho = True,
											beta = 0.5, c = 1e-4, tol = 1e-6, epsilon = 1e-8, maxiter = 10,
											cg_rtol = 1e-10, cg_atol = 1e-50, cg_maxiter = 1000,
											admm_tol = 1e-10, admm_maxiter = 200)
            
			t_CV = time.perf_counter() - t0
			print(f"{i}th, time: {t_CV:.3f} s")

			Sqs_CV = bft.get_Sqs(bft.kernel(qs_norm, thetas), weights, f_CV)
			history_list.append(history_CV)
			loss += torch.sum((Sqs_CV[:, i] - Sqs[:, i])**2)

			del f_CV, Sqs_CV, history_CV
			gc.collect()
			torch.cuda.empty_cache()

		loss_table[m, n] = loss
		print("+-----------------------------------------------------------------------------+")
		

state = {
    "alpha_list": alpha_list,
	"beta_list": beta_list,
	"loss_table": loss_table,
	"history_list": history_list}
	
torch.save(state, "/work/xunan/MaxEnt/Results/CV_beta0e0.pt")

