
import torch

from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import csr_matrix

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.linalg as cpx_linalg
from torch.utils.dlpack import to_dlpack, from_dlpack

import numpy as np
from dipy.segment.mask import median_otsu
from scipy import ndimage


def explain_this_module():
    """
    qs:         in shape (M, 2), where M is the number of 2D q points
    thetas:     in shape (N, 2), where N is the number of discretized 2D theta points
    weights:    in shape (N, ), the weights of the quadrature
    Sqs:        in shape (M, ), the observed signal at the q-vectors
    f0:         in shape (N, ), the demnominator in the KL divergence, default is None, which means all ones
    Lambdas:    in shape (M, ), the initial guess of dual variables, default is None, which means all ones
    scale:      a scaling factor for Sqs to avoid numerical issues (maybe), default is 1. It is some black magic coming from Lipeng Ning
    mu:         regularization parameter on Lambdas, default is 0. It makes the hessian positive definite
    beta:       step size reduction factor for Armijo line search, default is 0.5
    c:          constant for Armijo condition, default is 1e-4
    tol:        tolerance for convergence, default is 1e-5
    maxiter:    maximum number of iterations, default is 100
    f_true:     in shape (N, ), the true pdf for plotting purpose, default is None
    """
    return None

def Cartesian(theta1, theta2):
    """
    This function computes the Cartesian coordinates of two vectors.
    It's equivalent to generating a meshgrid and then flattening it.
    It also computes the weights of the quadrature.

    theta1: which is a vector of size (n1,), and it is in ascending order
    theta2: which is a vector of size (n2,), and it is in ascending order
    """

    theta11, theta22 = torch.meshgrid(theta1, theta2, indexing='ij')
    thetas = torch.stack([theta11.flatten(), theta22.flatten()], dim=1)             # in shape (n1*n2, 2)

    delta1 = torch.gradient(theta1)[0]                                              # in shape (n1,)
    delta2 = torch.gradient(theta2)[0]                                              # in shape (n2,)

    weights = torch.outer(delta1, delta2).reshape(-1) 

    return thetas, weights


def kernel(qs, thetas):
    """
    This function computes the kernel matrix for the given qs and thetas.
    """
    return torch.exp(- qs @ thetas.T)                                               # in shape (M, N). M = qs.shape[0], N = thetas.shape[0]

def get_Sqs(ker_mat, weights, f_thetas, sigma = 0):
    """
    This function computes the results of a forward model, that is, the y in y = Ax.

    ker_mat:        in shape (M, N)
    weights:        in shape (N, )
    f_thetas:       in shape (N, ) or shape (V, N)
    sigma:          noise level, a scalar, default is 0
    """
    size = ker_mat.shape[0] if f_thetas.ndim == 1 else (f_thetas.shape[0], ker_mat.shape[0])
    epsilon = torch.randn(size) * sigma
    return (f_thetas * weights) @ ker_mat.T + epsilon                               # in shape (M,) or shape (V, M)

#   ###############################################################################################
#   mainly designed for comparing two pdfs
#   ###############################################################################################

def compare_KL(f_num, f_den, weights):
    """
    This function computes the KL divergence between two pdfs and the orders matter.
    
    f_num:      in shape (N, ), the pdf in the numerator in the KL divergence
    f_den:      in shape (N, ), the pdf in the denominator in the KL divergence, no zero entries
    weights:    in shape (N, ), the weights of the quadrature
    """
    assert f_num.ndim == f_den.ndim, "Dimensions of f_num and f_den must match"
    return torch.sum(weights * f_num * torch.log(torch.maximum(f_num / torch.maximum(f_den, torch.tensor(1e-30)), torch.tensor(1e-30))),
                     dim=0 if f_num.ndim == 1 else 1)                               # a scalar if 1D or shape (V, ) if 2D

def compare_L2(f_a, f_b, weights=None):
    """
    This function computes the L2 distance between two pdfs and the orders do not matter.
    
    f_a:        in shape (N, ), the pdf in the first argument
    f_b:        in shape (N, ), the pdf in the second argument
    weights:    in shape (N, ), the weights of the quadrature. If None, it is assumed to be all ones.
    """
    if weights is None:
        weights = torch.ones_like(f_a[0])
    assert f_a.ndim == f_b.ndim, "Dimensions of f_a and f_b must match"
    return torch.sqrt(torch.sum(weights * (f_a - f_b) ** 2, dim=0 if f_a.ndim == 1 else 1))
    
#   ###############################################################################################
#   inverse or solve a linear system with a symmetric positive definite matrix
#   ###############################################################################################

def chol_solver(A, y=None):
    """
    This function solves the linear system Ax = y using Cholesky factorization.
    If y is None, it computes the inverse of A.
    Here the solver is designed for small dimension spd A.
    The form of A is dense.
    """
    A = A.to(torch.float64)      
    L = torch.linalg.cholesky(A)
    if y is None:
        return torch.cholesky_inverse(L)
    else:
        y = y.double().unsqueeze(-1)
        x = torch.cholesky_solve(y, L)
        return x.squeeze(-1)

def cupy_solver(A_csr, y, rtol=1e-10, atol=1e-50, maxiter=1000):
    """
    Solve A x = y for SPD matrix A (torch.sparse_csr_tensor on CUDA)
    using CuPy's conjugate gradient (CG) solver entirely on GPU.
    A_csr is high dimensional spd.
    """

    # --- Convert torch -> cupy (zero-copy via DLPack) ---
    data_cp    = cp.from_dlpack(to_dlpack(A_csr.values()))
    indices_cp = cp.from_dlpack(to_dlpack(A_csr.col_indices()))
    indptr_cp  = cp.from_dlpack(to_dlpack(A_csr.crow_indices()))
    y_cp       = cp.from_dlpack(to_dlpack(y))

    n = int(indptr_cp.size - 1)
    
    # --- Build CuPy CSR matrix ---
    A_cp = cpx_sparse.csr_matrix((data_cp, indices_cp, indptr_cp), shape=(n, n))

    # --- Simple Jacobi preconditioner (safe for SPD) ---
    D = A_cp.diagonal()
    D = cp.where(D == 0, 1.0, D)
    Minv = 1.0 / D

    def apply_M(x):
        return Minv * x

    M = cpx_linalg.LinearOperator((n, n), matvec=apply_M, dtype=A_cp.dtype)

    # --- Solve with Conjugate Gradient (CG) ---
    x_cp, info = cpx_linalg.cg(A_cp, y_cp, tol=rtol, atol=atol, maxiter=maxiter, M=M)

    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")

    # --- Convert back to torch (zero-copy if possible) ---
    try:
        # <-- use the snake_case to_dlpack() to avoid the deprecation warning
        x_torch = from_dlpack(x_cp.to_dlpack())
    except Exception:
        x_torch = torch.from_numpy(cp.asnumpy(x_cp)).to(device=A_csr.device, dtype=A_csr.dtype)

    return x_torch

def scipy_solver(A_csr, y, rtol=1e-10, atol = 1e-50, maxiter=1000):
    """
    Solve A x = y for SPD matrix A (torch.sparse_csr_tensor)
    using scipy's conjugate gradient (CG) solver entirely on CPU.
    A_csr is high dimensional spd.
    """
    crow = A_csr.crow_indices().cpu().numpy()
    col  = A_csr.col_indices().cpu().numpy()
    vals = A_csr.values().cpu().numpy()
    n = A_csr.size(0)

    A_scipy = csr_matrix((vals, col, crow), shape=(n, n))
    y_np = y.cpu().numpy()

    # --- Simple Jacobi preconditioner ---
    D = A_scipy.diagonal()
    D[D == 0] = 1.0
    Minv = 1.0 / D
    def apply_M(x):
        return Minv * x
    M = LinearOperator((n, n), matvec=apply_M, dtype=A_scipy.dtype)

    # --- solve Ax = b using Conjugate Gradient ---
    x_np, info = cg(A_scipy, y_np, rtol = rtol, atol = atol, maxiter = maxiter, M = M)

    # --- convert back to torch tensor ---
    x = torch.from_numpy(x_np).to(y.dtype).to(y.device)
    return x


#   ###############################################################################################
#   mask the real data
#   ###############################################################################################

def mask_brain(signal, median_radius = 1, numpass = 4, vol_idx = [0], least_size = 300, keep_top = None):
    """
    signal:                 is a 4D torch tensor data, the first 3D are physical space, the last one is q-space
    least_size:             drop reagions with size smaller than least_size 
    keep_top:               keep only top 2 or 3 big regions. If None, skip
    """
    signal = signal.detach().cpu().numpy()

    _, mask = median_otsu(  signal,
                            median_radius=median_radius,
                            numpass=numpass,
                            autocrop=False,
                            vol_idx=vol_idx)
    
    if least_size > 1 or keep_top is not None:                  # drop regions with size smaller than least_size 

        structure = ndimage.generate_binary_structure(3, 1)     # the last place can only be 1, 2, 3
                                                                # 1, 2, 3 corresponds to 6, 18, 26 respectively

        labels, nlb = ndimage.label(mask, structure=structure)  # identify the seperate regions of the mask, labeled with 0, 1, 2
        sizes = np.bincount(labels.ravel())                     # count the size of each region 0, region 1, region 2

        print("Number of initial valid regions:", nlb)
        print("Sizes of each regions (1st is background):", sizes)

        keep_labels = np.where(sizes >= least_size)[0]          # return a pool to decide which region are kept, like [0, 2]
        keep_labels = keep_labels[keep_labels != 0]             # drop the original False voxels

        if keep_top is not None:                                # keep only first keep_top regions
            keep_labels = np.where(np.isin(sizes, np.sort(sizes[keep_labels])[::-1][0:keep_top]))[0]

        print("Number of kept regions:", len(keep_labels) )
        print("Sizes of kept regions:", sizes[keep_labels])

        mask = np.isin(labels, keep_labels)                     # use the region pool to decide which voxels will be turned off    

    return torch.from_numpy(mask).to('cuda').bool()
    

def mask_to_lin2idx(mask):
    """
    This function was a part of mask_brain.
    Then I realized that sometimes I want to modify mask manually and then regenerate lin2idx.
    So, I make it an independent function.
    """

    mask = mask.detach().cpu().numpy()

    coords = np.argwhere(mask)
    N = coords.shape[0]
    shape = mask.shape

    # lin2idx: a 3D array storing the order of mask==True voxels in the coordinate system of the smallest cube

    lin2idx = -np.ones(shape, dtype=int)
    lin_inds = np.ravel_multi_index(coords.T, dims=shape, order='C')
    lin2idx_flat = lin2idx.ravel()
    lin2idx_flat[lin_inds] = np.arange(N)
    lin2idx = lin2idx_flat.reshape(shape)              

    return torch.from_numpy(lin2idx).to('cuda')






