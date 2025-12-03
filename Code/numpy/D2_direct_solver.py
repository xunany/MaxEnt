
import numpy as np
import scipy.sparse as sp

import sys
import importlib
from pathlib import Path
sys.path.append(str(Path.home() / "Projects" / "MaxEnt" / "Code"))
import Basic_functions
importlib.reload(Basic_functions)
from Basic_functions import kernel, get_Sqs, compare_KL, chol_solver, PETSc_solver
from D2_admm_solver import data_check, loss_function, f_thetas_hat, negative_dual_function

def gradient_hessian(Lambdas, ker_mat, ker_prod, weights, Sqs, sigma, R_csr, f0, normalize, rtol, atol, maxiter):
    """
    This function computes the gradient and the hessian matrix for the negative dual function.

    Lambdas:        in shape (V, M)
    ker_mat:        in shape (M, N)
    ker_prod:       in shape (M, M, N)
    weights:        in shape (N, )
    Sqs:            in shape (V, M)
    f0:             in shape (V, N)
    """
    V = Sqs.shape[0]        # number of voxels
    M = ker_mat.shape[0]    # number of qs
    N = ker_mat.shape[1]    # number of thetas
        
    rSqs = Sqs.ravel(order='C')                                                                 # in shape (V*M, )
    rLambdas = Lambdas.ravel(order='C')                                                         # in shape (V*M, )

    Sigma_inv_diag = 1 / (sigma**2) * np.ones(V*M)                                              # in shape (V*M, )
    middle_matrix = sp.diags(Sigma_inv_diag, format='csr') + R_csr                              # in shape (V*M, V*M)
    side_vector = Sigma_inv_diag * rSqs + rLambdas                                              # in shape (V*M, )

    Sigma_diag = (sigma**2) * np.ones(V*M)                                                      # in shape (V*M, )

    f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize = normalize)                  # in shape (V, N)
    f_weights = f_hat * weights                                                                 # in shape (V, N)
    grad1 = f_weights @ ker_mat.T                                                               # in shape (V, M)
    if (R_csr.nnz == 0):
        grad2 = rSqs + np.diag(Sigma_diag) @ rLambdas
    else:
        grad2 = PETSc_solver(middle_matrix, side_vector, rtol, atol, maxiter)                   # in shape (V*M, )
    gradient = - grad1.ravel(order='C') + grad2                                                 # in shape (V*M, )

    hess_blocks = np.tensordot(f_weights, ker_prod, axes=([1], [2]))                            # in shape (V, M, M)
    hess = sp.block_diag([hess_blocks[i] for i in range(V)], format='csr')                      # in shape (V*M, V*M)

    if normalize:
        hess = hess - sp.csr_matrix(np.outer(grad1.ravel(order='C'), grad1.ravel(order='C')))   # in shape (V*M, V*M)
    if (R_csr.nnz == 0):
        hess = hess + sp.diags(Sigma_diag, format = 'csr')                                      # in shape (V*M, V*M)
    else:
        hess = hess + sp.csr_matrix(chol_solver(middle_matrix.toarray()))                       # in shape (V*M, V*M)
    return gradient, hess

def Newton_Armijo(  qs, thetas, weights, Sqs, sigma, R_csr, f0 = None, normalize = False,
                    Lambdas = None, 
                    beta = 0.5, c = 1e-4, epsilon = 1e-8, tol = 1e-6, maxiter = 100,
                    cg_rtol = 1e-10, cg_atol = 1e-50, cg_maxiter = 1000):
    """
    This function implements the Newton's method with Armijo line search to optimize the dual variables Lambdas.

    qs:         in shape (M, 2)
    thetas:     in shape (N, 2)
    weights:    in shape (N, )
    Sqs:        in shape (V, M)
    sigma:      noise level, the standard deviation of the noise
    R:          in shape (V*M, V*M), the roughness regularization matrix
    f0:         in shape (V, N)

    Lambdas:    in shape (V, M), initial guess for the dual variables Lambdas

    beta:       step size reduction factor for Armijo line search, default is 0.5
    c:          constant for Armijo condition, default is 1e-4
    epsilon:    the smallest value the step can be, default is 1e-8
    tol:        tolerance for convergence, default is 1e-6
    max_iter:   maximum number of iterations, default is 100
    stabilize:  whether to add a small value to the hessian matrix in the first few iterations. Default is True
    """

    V = Sqs.shape[0] if Sqs.ndim == 2 else 1            # number of voxels
    M = qs.shape[0]                                     # number of qs
    N = thetas.shape[0]                                 # number of thetas

    if f0 is None:
        f0 = 1/np.sum(weights)*np.ones((V, N))          # in shape (V, N)

    qs, thetas, weights, Sqs, sigma, R_csr, f0, _ = data_check(qs, thetas, weights, Sqs, sigma, R_csr, f0)

    ker_mat = kernel(qs, thetas)                        # in shape (M, N)
    ker_prod = ker_mat[:, None, :] * ker_mat[None, :, :]# in shape (M, M, N) mn,in->min

    if Lambdas is None:
        Lambdas = np.ones((V, M))                       # in shape (V, M)

    rLambdas = Lambdas.ravel(order='C')                 # in shape (V*M, )

    loop = 0

    obj_history = []
    obj_current = negative_dual_function(   Lambdas, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize,
                                            cg_rtol, cg_atol, cg_maxiter)  
    obj_history.append(obj_current)

    while True:
        
        grad, hess = gradient_hessian(  Lambdas, ker_mat, ker_prod, weights, Sqs, sigma, R_csr, f0, normalize, 
                                        cg_rtol, cg_atol, cg_maxiter)                           # in shape (V*M, ), (V*M, V*M)
        
        vals = np.sort(sp.linalg.eigsh(hess, k=24)[0])
        print(vals[0], vals[-1]) 

        direction = - PETSc_solver(hess, grad, cg_rtol, cg_atol, cg_maxiter)

        dot_product = grad @ direction                                                          # a scalar

        if -dot_product <= tol:
            break

        step = 1.0
        while True:
            rcandidate = rLambdas + step * direction                                            # in shape (V*M, )
            candidate = rcandidate.reshape(V, M, order='C')                                     # in shape (V, M)
            obj_new = negative_dual_function(   candidate, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize,
                                                cg_rtol, cg_atol, cg_maxiter) 

            if obj_new <= obj_current + c * step * dot_product:
                break

            step *= beta
            if step < epsilon:                                                                  # small step size, stop shrinking
                break

        # Update
        rLambdas = rLambdas + step * direction
        Lambdas = rLambdas.reshape(V, M, order='C')                                             # in shape (V, M)
        
        obj_current = negative_dual_function(   Lambdas, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize,
                                                cg_rtol, cg_atol, cg_maxiter) 
        obj_history.append(obj_current)

        loop += 1
        if loop >= maxiter:
            print(f"Newton Armijo maximum iterations reached: {maxiter}")
            break                                

    f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize)                              # Update the pdf estimate
    return Lambdas, f_hat, obj_history


