
import torch

import sys
import importlib
sys.path.append("/work/xunan/MaxEnt/Code")
import Basic_functions_torch
importlib.reload(Basic_functions_torch)
from Basic_functions_torch import kernel, get_Sqs, cupy_solver, PETSc_solver, compare_KL # 

def data_check(qs, thetas, weights, Sqs, sigma, R_csr, f0, f_hat = None):
    """
    This function checks the data types and shapes of the input parameters.
    """
    assert isinstance(qs, torch.Tensor), "qs must be a torch Tensor"
    assert isinstance(thetas, torch.Tensor), "thetas must be a torch Tensor"
    assert isinstance(weights, torch.Tensor), "weights must be a torch Tensor"
    assert isinstance(Sqs, torch.Tensor), "Sqs must be a torch Tensor"
    assert isinstance(sigma, torch.Tensor), "sigma must be a torch Tensor, even though it can be a scalar"
    assert isinstance(R_csr, torch.Tensor) and R_csr.layout == torch.sparse_csr, "R must be a torch.sparse_csr_tensor"
    assert isinstance(f0, torch.Tensor), "f0 must be a torch Tensor"
    
    if Sqs.ndim == 1:
        Sqs = Sqs[None, :]                              # For a single voxel, Sqs is a 1D array, we need to make it 2D

    if f0.ndim == 1:
        f0 = f0[None, :]                                # the same reason as above, we need to make it 2D

    assert qs.ndim == 2 and qs.shape[1] == 2, "qs must be of shape (M, 2)"

    if sigma.ndim == 0:
        sigma = sigma * torch.ones(Sqs.shape[0] * qs.shape[0])
    if sigma.ndim != 0:
        if not (sigma.ndim == 1 and sigma.numel() == Sqs.shape[0] * qs.shape[0]):
            raise ValueError("Size of sigma is not correct.")

    assert thetas.ndim == 2 and thetas.shape[1] == 2, "thetas must be of shape (N, 2)"
    assert weights.ndim == 1 and weights.shape[0] == thetas.shape[0], "weights must match the number of thetas"
    assert Sqs.shape[1] == qs.shape[0], "Sqs must match the number of qs"
    assert f0.shape[0] == Sqs.shape[0] and f0.shape[1] == thetas.shape[0], "f0 shape mismatch"

    if f_hat is not None:
        assert isinstance(f_hat, torch.Tensor), "f_hat must be a numpy array"
        if f_hat.ndim == 1:
            f_hat = f_hat[None, :]                      # the same reason as above, we need to make it 2D
        assert f_hat.shape[0] == Sqs.shape[0] and f_hat.shape[1] == thetas.shape[0], "f_hat shape mismatch"

    if not (R_csr.shape[0] == R_csr.shape[1] and R_csr.shape[0] == Sqs.shape[0] * qs.shape[0]):
        raise ValueError("Size of R_csr is not correct.")

    return qs, thetas, weights, Sqs, sigma, R_csr, f0, f_hat

def loss_function(qs, thetas, weights, Sqs, R_csr, f_hat, sigma, f0):
    """
    Torch version of the original loss function for the D2 multiple voxels model
    (WITH noise and WITH roughness). Returns the three terms of the loss.
    """
    qs, thetas, weights, Sqs, sigma, R_csr, f0, f_hat = data_check(qs, thetas, weights, Sqs, sigma, R_csr, f0, f_hat)

    V = Sqs.shape[0]                                                            # number of voxels
    M = qs.shape[0]                                                             # number of qs
    N = thetas.shape[0]                                                         # number of thetas

    ker_mat = kernel(qs, thetas)

    first_term = torch.sum(compare_KL(f_hat, f0, weights))                      # a scalar                                              

    Sqs_hat = get_Sqs(ker_mat, weights, f_hat)                                  # shape (V, M)

    rSqs = Sqs.reshape(-1)                                                      # shape (V*M,)
    rSqs_hat = Sqs_hat.reshape(-1)                                              # shape (V*M,)
    side_vector = rSqs - rSqs_hat                                               # shape (V*M,)
    second_term = 0.5 * (torch.linalg.norm(side_vector) ** 2) / (sigma ** 2)    # a scalar

    third_term = 0.5 * (rSqs_hat @ (R_csr @ rSqs_hat))                          # a scalar

    return first_term, second_term, third_term                                  # three scalars

# def f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize):

#     """
#     This function computes the pdf hat{f}_thetas given the dual variables Lambdas.

#     Lambdas:        in shape (V, M)
#     ker_mat:        in shape (M, N)
#     weights:        in shape (N, )
#     f0:             in shape (V, N)
#     """
#     V = Lambdas.shape[0]        # number of voxels
#     N = ker_mat.shape[1]        # number of thetas

#     f_hat = f0 * torch.exp(torch.clamp(- Lambdas @ ker_mat - 1, min=-700, max=700))     # in shape (V, N)

#     if normalize:
#         f_hat /= ((f_hat @ weights).unsqueeze(1).repeat(1, N))                          # normalize the pdf
#     return f_hat

def f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize):
    """
    Numerically-stable computation of f_hat = f0 * exp(-Lambda @ K - 1) normalized
    by the weighted sum with `weights`. Works voxel-wise.

    By the way, I have to admit that it's ChatGPT who modifies this function.
    It considers some underflow/overflow issues in my original function. 

    Lambdas:        in shape (V, M)
    ker_mat:        in shape (M, N)
    weights:        in shape (N, )
    f0:             in shape (V, N)
    """

    eps = 1e-300                                            # set a criterion to avoid log(0). No need to change it

    V = Lambdas.shape[0]
    N = ker_mat.shape[1]

    if not normalize:
        f_hat = f0 * torch.exp(torch.clamp(-(Lambdas @ ker_mat) - 1.0, min=-700, max=700))
        return f_hat
    
    else:                                                   # if normalize, we need to calculate the integral and might lead to overflow issues
        log_f0 = torch.log(torch.clamp(f0, min=eps))        # in shape (V, N)
        log_f_hat = log_f0 - (Lambdas @ ker_mat) - 1.0      # shape (V, N)

        row_max = torch.max(log_f_hat, dim=1, keepdim=True).values  # in shape (V, 1) Return the per-row max
        exp_shift = torch.exp(log_f_hat - row_max)          # in shape (V, N) Subtract per-row max for stability

        denom = torch.clamp(exp_shift @ weights, min=eps)   # in shape (V, )  Calculate the integral after exp shifting. Clip it if too small
        f_hat = exp_shift / denom.unsqueeze(1).repeat(1, N) # in shape (V, N)
        return f_hat

def negative_dual_function(Lambdas, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize, rtol, atol, maxiter):
    """
    This function defines the objective function to be optimized.
    Attention, this function needs to be MINIMIZED to optimize Lambdas!

    Lambdas:        in shape (V, M)
    ker_mat:        in shape (M, N)
    weights:        in shape (N, )
    Sqs:            in shape (V, M)
    sigma:          a scalar or a vector. Which is the noise level, the standard deviation of the noise
    R:              in shape (V*M, V*M). The roughness regularization parameter
    f0:             in shape (V, N)
    """
    V = Sqs.shape[0]                                                                    # number of voxels
    M = ker_mat.shape[0]                                                                # number of qs
    N = ker_mat.shape[1]                                                                # number of thetas
    
    rSqs = Sqs.reshape(-1)                                                              # in shape (V*M, )
    rLambdas = Lambdas.reshape(-1)                                                      # in shape (V*M, )

    nodes = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize=False)                # in shape (V, N)
    if normalize:
        first_term = torch.sum(torch.log(nodes @ weights))                              # a scalar
    else:
        first_term = torch.sum(nodes @ weights)                                         # a scalar
    
    if R_csr.values().numel() == 0:
        second_term = torch.dot(rSqs, rLambdas) + 0.5 * torch.linalg.norm(rLambdas * sigma) ** 2                            # a scalar
    else:
        Sigma_inv_diag = (1 / (sigma ** 2)) * torch.ones(V * M)
        middle_matrix = torch.sparse_csr_tensor(
                            torch.arange(V*M + 1),
                            torch.arange(V*M),
                            (1.0 / (sigma**2)),
                            size=(V*M, V*M)  ) + R_csr                                  # in shape (V*M, V*M)
        side_vector = Sigma_inv_diag * rSqs + rLambdas                                  # in shape (V*M, )
        second_term = 0.5 * (side_vector @ cupy_solver(middle_matrix, side_vector, rtol=rtol, atol = atol, maxiter=maxiter)) # a scalar
    
    return first_term + second_term

def gradient_hessian(Lambdas, ker_mat, ker_prod, weights, Sqs, sigma, R_csr, f0, normalize, rtol, atol, maxiter, blocked = False):
    """
    This function computes the gradient and the hessian matrix for the negative dual function.
    It has no use to solving ADMM.
    It can be used in returning the blocked hessian matrix given the optimum of Lambdas,
    which is needed in generalized corss validation.

    Lambdas:    in shape (V, M)
    ker_mat:    in shape (M, N)
    ker_prod:   in shape (M, M, N)
    weights:    in shape (N, )
    Sqs:        in shape (V, M)
    f0:         in shape (V, N)

    blocked:    if True, only return blocked hess by assuming Lambdas is already optimal and gradient nearly == 0.
                And no matter what R_csr is, it will be treated as all zeros, i.e., no spatial smoothness is considered.
                blocked_hess is meaningful only if the optimizer is solved with R_csr == 0.
                However, it still can be returned even if not. Keep that in mind and be careful.
    """
    V = Sqs.shape[0]        # number of voxels
    M = ker_mat.shape[0]    # number of qs
    N = ker_mat.shape[1]    # number of thetas
        
    Sigma_diag = (sigma**2) * torch.ones(V*M)                                               # in shape (V*M, )

    f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize = normalize)              # in shape (V, N)
    f_weights = f_hat * weights.view(1, -1)                                                 # in shape (V, N)
    grad1 = - f_weights.matmul(ker_mat.T)                                                   # in shape (V, M)

    hess_blocked = torch.tensordot(f_weights, ker_prod, dims=([1], [2]))                    # in shape (V, M, M)

    if normalize:
        hess_blocked -= grad1[:, :, None] * grad1[:, None, :]                               # in shape (V, M, M)

    if blocked:
        hess_blocked += torch.eye(M)[None, :, :] * Sigma_diag.reshape(V, M)[:, :, None]     # in shape (V, M, M)
        return hess_blocked
    
    # rSqs = Sqs.reshape(-1)                                                                  # in shape (V*M, )
    # rLambdas = Lambdas.reshape(-1)                                                          # in shape (V*M, )

    # Sigma_inv_diag = (1 / (sigma ** 2)) * torch.ones(V * M)                                 # in shape (V*M, )
    
    # middle_matrix = torch.sparse_csr_tensor(
    #                     torch.arange(V*M + 1),
    #                     torch.arange(V*M),
    #                     (1.0 / (sigma**2)),
    #                     size=(V*M, V*M)  ) + R_csr                                          # in shape (V*M, V*M)
    # side_vector = Sigma_inv_diag * rSqs + rLambdas                                          # in shape (V*M, )
    
    # if R_csr.values().numel() == 0:
    #     grad2 = rSqs + Sigma_diag * rLambdas                                                # in shape (V*M, )
    # else:
    #     grad2 = cupy_solver(middle_matrix, side_vector, rtol, atol, maxiter)                # in shape (V*M, )

    # gradient = - grad1.reshape(-1) + grad2                                                  # in shape (V*M, )

    # hess = torch.block_diag(*[hess_blocked[i] for i in range(V)])                           # in shape (V*M, V*M)

    # indptr = torch.arange(0, V * M * M + 1, M)
    # row_ids = torch.arange(V * M)
    # indices = (row_ids // M)[:, None] * M + torch.arange(M)[None, :]

    # values = hess_blocked[ row_ids // M, row_ids % M, torch.arange(M)[None, :] ]

    # hess = torch.sparse_csr_tensor(
    #     indptr,
    #     indices.reshape(-1),
    #     values.reshape(-1),
    #     size=(V * M, V * M) )                                                               # in shape (V*M, V*M)

    # if R_csr.values().numel() == 0:
    #     hess = hess + torch.diag(Sigma_diag)                                                # in shape (V*M, V*M)
    # else:
    #     hess = hess + cupy_solver(middle_matrix)                                            # in shape (V*M, V*M)

    # return gradient, hess

def admm(   qs, thetas, weights, Sqs, sigma, R_csr, f0 = None, normalize = True, Lambdas = None, 
            beta = 0.5, c = 1e-4, epsilon = 1e-8, tol = 1e-6, maxiter = 100, 
            cg_rtol = 1e-10, cg_atol = 1e-50, cg_maxiter = 1000, 
            rho = 1.0, rho_ratio = 3, dynamic_rho = False,
            admm_tol = 1e-8, admm_maxiter = 100,
            return_hess = False):
    
    """
    rho_ratio:  should be strictly greater than 1
    keep_hess:  If use generalized cross validation, hess needs to be returned.
                If do voxel-wise gcv for just alpha(s), set R_csr to empty please, and it returns "blocked_hess".
                If do gcv for both (alpha, if needed) beta, no requirement to R_csr, and it still returns "blocked_hess",
                but notice that "blocked hess" is not the final hess in that case.
    """
    
    def dist_obj(Lambdas, zs, us, rho, ker_mat, weights, f0, normalize):
        """
        Here the dist means distributed.
        Lambdas : in shape (V, M)
        zs      : in shape (V, M)
        us      : in shape (V, M)
        """

        nodes = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize=False)                # in shape (V, N)
        if normalize:
            first_term = torch.log(nodes @ weights)                                         # in shape (V, )
        else:
            first_term = nodes @ weights                                                    # in shape (V, )
                        
        second_term = 0.5 * rho * torch.linalg.norm(Lambdas - zs + us, dim=1)**2            # in shape (V, )

        return first_term + second_term                                                     # in shape (V, )

    def dist_grad_hess(Lambdas, zs, us, rho, ker_mat, ker_prod, weights, f0, normalize):
        """
        Here the dist means distributed.
        Lambdas : in shape (V, M)
        zs      : in shape (V, M)
        us      : in shape (V, M)
        """

        M = ker_mat.shape[0]    # number of qs

        f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize = normalize)      # in shape (V, N)
        f_weights = f_hat * weights.view(1, -1)                                         # in shape (V, N)
        grad = - f_weights.matmul(ker_mat.T) + rho * (Lambdas - zs + us)                # in shape (V, M)

        hess = torch.tensordot(f_weights, ker_prod, dims=([1],[2]))                     # in shape (V, M, M) vn, ijn -> vij
        I = torch.eye(M).view(1, M, M)
        hess = hess + rho * I

        if normalize:
            tmp = f_weights.matmul(ker_mat.T)                                           # in shape (V, M)
            hess = hess - tmp.view(V, M, 1) @ tmp.view(V, 1, M)                         # in shape (V, M, M)

        return grad, hess

    def dist_Newton_Armijo( Lambdas, zs, us, rho, ker_mat, ker_prod, weights, f0, normalize,
                            beta, c, epsilon, tol, maxiter):
        """
        Here the dist means distributed.
        Lambdas : in shape (V, M)
        zs      : in shape (V, M)
        us      : in shape (V, M)
        """

        V = Lambdas.shape[0]
        
        obj_current = dist_obj(Lambdas, zs, us, rho, ker_mat, weights, f0, normalize)                       # in shape (V, )
        active = torch.ones(V, dtype=torch.bool)                                                            # in shape (V, )

        loop = 0

        iter_history = torch.zeros(V, dtype=bool)                                   # to store if Newton works in each voxel

        while True:

            grad, hess = dist_grad_hess(Lambdas, zs, us, rho, ker_mat, ker_prod, weights, f0, normalize)
            direction = - torch.cholesky_solve(grad.unsqueeze(-1), torch.linalg.cholesky(hess)).squeeze(-1) # in shape (V, M)

            dot_product = (grad * direction).sum(dim=1)                                                     # in shape (V, )

            converged = ( -dot_product <= tol )
            active = active & (~converged)
            if not active.any().item():
                break
                   
            step = torch.ones(V)                                                                            # in shape (V, )

            searching = active.clone()

            while True:
                if not searching.any().item():
                    break

                candidate = Lambdas + step.view(V, 1) * direction                                           # in shape (V, M)
                obj_new = dist_obj(candidate, zs, us, rho, ker_mat, weights, f0, normalize)                 # in shape (V, )

                satisfied = obj_new <= (obj_current + c * step * dot_product)                               # in shape (V, )

                searching = searching & (~satisfied)

                step = torch.where(searching, step * beta, step) 
                
                give_up = (step < epsilon) & searching       
                if give_up.any():
                    searching = searching & (~give_up)

            Lambdas = Lambdas + step.view(V, 1) * direction

            obj_current = dist_obj(Lambdas, zs, us, rho, ker_mat, weights, f0, normalize)                   # in shape (V, )

            loop += 1
            iter_history[active] = True
            if loop >= maxiter:
                break

        return Lambdas, iter_history
    
    """
    start -------------------------------------------------------------------------------------------------------------
    """

    V = Sqs.shape[0] if Sqs.ndim == 2 else 1                                                    # number of voxels
    M = qs.shape[0]                                                                             # number of qs
    N = thetas.shape[0]                                                                         # number of thetas

    if f0 is None:
        f0 = (1.0 / torch.sum(weights)) * torch.ones((V, N))                                    # in shape (V, N)

    qs, thetas, weights, Sqs, sigma, R_csr, f0, _ = data_check(qs, thetas, weights, Sqs, sigma, R_csr, f0)

    ker_mat = kernel(qs, thetas)                                                                # in shape (M, N)
    ker_prod = ker_mat[:, None, :] * ker_mat[None, :, :]                                        # in shape (M, M, N) mn,in->min

    if Lambdas is None:
        Lambdas = torch.ones((V, M))                                                            # in shape (V, M)

    zs = torch.ones(V*M)                                                                        # in shape (V*M, )
    us = torch.ones(V*M)                                                                        # in shape (V*M, )

    Sigma_inv = torch.sparse_csr_tensor(
                torch.arange(V*M + 1),
                torch.arange(V*M),
                (1.0 / (sigma**2)),
                size=(V*M, V*M)     )

    loop = 0

    obj_history = []
    obj_history.append(
        negative_dual_function(Lambdas, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize, cg_rtol, cg_atol, cg_maxiter)
        + (rho/2) * torch.linalg.norm(Lambdas.reshape(-1) - zs + us)**2
        - (rho/2) * torch.linalg.norm(us)**2
    )

    primal_history = []
    dual_history = []
    rho_history = []
    Newton_history = torch.zeros((V, admm_maxiter), dtype=bool)         # to store if Newton works in each voxel for each loop

    while True:

        """
        update Lambdas ------------------------------------------------------------------------------------------------
        """
            
        Lambdas, Newton_history[:, loop]= dist_Newton_Armijo(   Lambdas, zs.view(V, M), us.view(V, M), rho, ker_mat, ker_prod, 
                                                                weights, f0, normalize, beta, c, epsilon, tol, maxiter  )
        
        obj_history.append(
            negative_dual_function(Lambdas, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize, cg_rtol, cg_atol, cg_maxiter)
            + (rho/2) * torch.linalg.norm(Lambdas.reshape(-1) - zs + us)**2
            - (rho/2) * torch.linalg.norm(us)**2
        )

        """
        update zs -----------------------------------------------------------------------------------------------------
        """

        if R_csr.values().numel() == 0:
            vec_y = rho / (sigma**2) * (Lambdas.reshape(-1) + us) - ( Sqs.reshape(-1) / (sigma**2) )
            zs_new = vec_y / ( torch.tensor(1.0) + rho / (sigma**2) )

        else:

            mat_A = torch.sparse_csr_tensor(
                        torch.arange(V*M + 1), 
                        torch.arange(V*M),     
                        torch.ones(V*M),             
                        size=(V*M, V*M)    ) + rho * (Sigma_inv + R_csr)
            vec_y = rho * (Sigma_inv + R_csr) @ (Lambdas.reshape(-1) + us) - (Sigma_inv @ Sqs.reshape(-1))

            zs_new = cupy_solver(mat_A, vec_y, rtol=cg_rtol, atol=cg_atol, maxiter=cg_maxiter)

        """
        check convergence ---------------------------------------------------------------------------------------------
        """
        loop += 1

        primal = torch.linalg.norm(Lambdas.reshape(-1) - zs_new)
        dual   = torch.linalg.norm(zs_new - zs)

        primal_history.append(primal.item() if isinstance(primal, torch.Tensor) else primal)
        dual_history.append(dual.item() if isinstance(dual, torch.Tensor) else dual)

        if (primal <= admm_tol) and (dual <= admm_tol):
            break

        if loop >= admm_maxiter:
            print("ADMM maximum iterations reached.")
            break

        zs = zs_new

        """
        modify --------------------------------------------------------------------------------------------------------
        """
        if dynamic_rho:
            if primal > rho_ratio * dual:
                rho = rho * 2
                us = us / 2

            if dual > rho_ratio * primal:
                rho = rho / 2
                us = us * 2

        rho_history.append(rho)

        """
        update us -----------------------------------------------------------------------------------------------------
        """

        us = us + Lambdas.reshape(-1) - zs

        """
        end -----------------------------------------------------------------------------------------------------------
        """

    f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize)

    Newton_history = Newton_history[:, :loop]               # Unused part of Newton_history will be cropped

    history = [obj_history, primal_history, dual_history, rho_history, Newton_history]

    if return_hess:
        final_blocked_hess = gradient_hessian(  Lambdas, ker_mat, ker_prod, weights, Sqs, sigma, R_csr, 
                                                f0, normalize, cg_rtol, cg_atol, cg_maxiter, blocked = True)
        history.append(final_blocked_hess)
    
    return Lambdas, f_hat, history                                

def uncertainty(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize):
    qs, thetas, weights, Sqs, sigma, R, f0, _ = data_check(qs, thetas, weights, Sqs, sigma, R, f0)

    device = Sqs.device if hasattr(Sqs, "device") else torch.device("cpu")
    dtype  = Sqs.dtype  if hasattr(Sqs, "dtype")  else torch.get_default_dtype()

    V = Sqs.shape[0]
    M = Sqs.shape[1]
    N = thetas.shape[0]

    ker_mat = kernel(qs, thetas)                                                                                    # in shape (M, N)
    ker_prod = torch.einsum('ik,jk->ijk', ker_mat, ker_mat)                                                         # in shape (M, M, N)
    f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize = normalize)                                     # in shape (V, N)

    Gamma = torch.zeros((V*M, V*M), device=device, dtype=dtype)                                                     # in shape (V*M, V*M)
    for i in range(V):
        block = torch.einsum('ijn,n->ij', ker_prod, (f_hat[i, :] * weights))                                        # in shape (M, M)
        Gamma[i*M:(i+1)*M, i*M:(i+1)*M] = block                                                                      # hess block

    if normalize:
        grad = torch.einsum('vn,mn->vmn', f_hat, ker_mat)                                                           # in shape (V, M, N)
        grad = torch.einsum('vmn,n->vm', grad, weights)                                                             # in shape (V, M)
        g_col = grad.reshape(-1, 1)                                                                                 # (V*M, 1)
        g_row = grad.reshape(1, -1)                                                                                 # (1, V*M)
        Gamma = Gamma - (g_col @ g_row)                                                                             # in shape (V*M, V*M)
    
    Sigma_inv = torch.eye(V*M, device=device, dtype=dtype) * (1.0 / (sigma**2))                                    # in shape (V*M, V*M)
    Sigma     = torch.eye(V*M, device=device, dtype=dtype) * (sigma**2)                                              # in shape (V*M, V*M)

    precision = ( Gamma @ (Sigma_inv + 2.0 * R + R @ Sigma @ R) @ Gamma
                 + 2.0 * Gamma @ (torch.eye(V*M, device=device, dtype=dtype) + R @ Sigma)
                 + Sigma )                                                                                         # in shape (V*M, V*M)

    # # ---------------------------------------------------------------------------------------------
    # # From the important truth that R = 0 !
    #
    # part = inverse_spd(Gamma + Sigma)
    # fir_der = - part
    # # ---------------------------------------------------------------------------------------------
    #
    # ker_prod2 = torch.einsum('ik,jk,mk->ijmk', ker_mat, ker_mat, ker_mat)                                          # in shape (M, M, M, N)
    # Third_D = torch.zeros((V*M, V*M, V*M), device=device, dtype=dtype)                                             # in shape (V*M, V*M, V*M)
    # for i in range(V):
    #     Third_D[i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M] = torch.einsum('ijkn,n->ijk', ker_prod2, f_hat[i, :] * weights) # in shape (V*M, V*M, V*M)
    #            
    # sec_der = torch.einsum('ijk, ia, jb, kc -> abc', Third_D, part, part, part)
    # # ---------------------------------------------------------------------------------------------
    #
    # ker_prod3 = torch.einsum('ik,jk,mk,lk->ijmlk', ker_mat, ker_mat, ker_mat, ker_mat)                               # in shape (M, M, M, N)
    # Fourth_D = torch.zeros((V*M, V*M, V*M, V*M), device=device, dtype=dtype)
    # for i in range(V):
    #     Fourth_D[i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M] = torch.einsum('ijkln,n->ijkl', ker_prod3, f_hat[i, :] * weights)
    #
    # thi_der = (     torch.einsum('ijkl, ia, jb, kc, ld -> abcd', Fourth_D, part, part, part, part)
    #             -   3 * torch.einsum('ai, ijk, jbc, kd -> abcd', part, Third_D, sec_der, part)     )
    # # ---------------------------------------------------------------------------------------------

    return precision    #, fir_der, sec_der, thi_der


