import torch
import time
import numpy as np

import nibabel as nib
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table

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

data = np.load('/work/xunan/MaxEnt/Data/Processed_Ning.npz')

Sb0 = torch.from_numpy(data['Sb0']).to('cuda')
Sqs = torch.from_numpy(data['Sqs']).to('cuda')
qs  = torch.from_numpy(data['qs'] ).to('cuda')

###########################################
# Identify the white matter
###########################################

data = nib.load("/work/xunan/MaxEnt/Data/MultiTE_dMRI/TE71ms/Processed/dwi_eddy_unring.nii").get_fdata()	# read TE71ms raw data
bvals = np.loadtxt("/work/xunan/MaxEnt/Data/MultiTE_dMRI/TE71ms/Processed/dwi_bval.txt")					# read b-values
bvecs = np.loadtxt("/work/xunan/MaxEnt/Data/MultiTE_dMRI/TE71ms/Processed/dwi_ed_bvec.txt")					# read b-vectors

gtab = gradient_table(bvals=bvals, bvecs=bvecs)

tenmodel = TensorModel(gtab)																				# Fit tensor model
tenfit = tenmodel.fit(data)

FA = tenfit.fa

white_mask = FA > 0.2																						# recommended value 0.2-0.3
white_mask = torch.from_numpy(white_mask).to('cuda')

###########################################
# Process the data
###########################################

brain_mask = bft.mask_brain(Sb0, median_radius = 1, numpass = 4, vol_idx = [0, 1, 2, 3, 4], least_size = 100, keep_top = None)
mask = brain_mask & white_mask
lin2idx = bft.mask_to_lin2idx(mask)

theta1 = torch.linspace(0, 3, 50)
theta2 = torch.linspace(0, 30, 50)

thetas, weights = bft.Cartesian(theta1, theta2)

R0 = torch.sparse_csr_tensor(
    torch.zeros(torch.sum(mask)*25 + 1),
    torch.empty(0),
    torch.empty(0),
    size=(torch.sum(mask)*25, torch.sum(mask)*25),
    device='cuda'
)

###########################################
# Set R to zero to estimate noise level
###########################################

print("+-----------------------------------------------------------------------------+")

std_vector = 0.05 * torch.ones(len(Sqs[mask, :]))
sigma = std_vector.repeat_interleave(25)

for i in range(3):
	
	t0 = time.perf_counter()

	_, f_est, _ = dast.admm(	qs, thetas, weights, Sqs[mask, :], sigma = sigma,
								R_csr = R0, f0 = None, normalize = True, Lambdas = None, 
								beta = 0.5, c = 1e-4, epsilon = 1e-8, tol = 1e-12, maxiter = 1,
								cg_rtol = 1e-10, cg_atol = 1e-50, cg_maxiter = 1000,
								rho = 0.5, rho_ratio = 2, dynamic_rho = True,
								admm_tol = 1e-10, admm_maxiter = 200)
	
	t_est = time.perf_counter() - t0
	print(f"Pre-estimation: {t_est:.3f} s")

	diff = Sqs[mask, :] - bft.get_Sqs(bft.kernel(qs, thetas), weights, f_est)
	std_vector = diff.std(dim=1)
	sigma = std_vector.repeat_interleave(25)

print("+-----------------------------------------------------------------------------+")
print("estimation of sigma is:", torch.mean(sigma))

state = {
	"Sb0": Sb0,
    "Sqs": Sqs,
	"qs": qs,
	"theta1": theta1,
	"theta2": theta2,
	"thetas": thetas,
	"weights": weights,
	"mask": mask,
	"lin2idx": lin2idx,
	"std_vector": std_vector,
	"sigma": sigma				}
	
torch.save(state, "/work/xunan/MaxEnt/Data/Sigma_estimated.pt")



