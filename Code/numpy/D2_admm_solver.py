import numpy as np
import scipy.sparse as sp

from Basic_functions import kernel, get_Sqs, chol_solver, PETSc_solver, compare_KL

def data_check(qs, thetas, weights, Sqs, sigma, R_csr, f0, f_hat = None):
    """
    This function checks the data types and shapes of the input parameters.
    """
    assert isinstance(qs, np.ndarray), "qs must be a numpy array"
    assert isinstance(thetas, np.ndarray), "thetas must be a numpy array"
    assert isinstance(weights, np.ndarray), "weights must be a numpy array"
    assert isinstance(Sqs, np.ndarray), "Sqs must be a numpy array"
    assert isinstance(sigma, (int, float, np.ndarray)), "sigma must be a scalar or a numpy array"
    assert isinstance(R_csr, sp.csr_matrix), "R_csr must be a spicy csr sparse matrix"
    assert isinstance(f0, np.ndarray), "f0 must be a numpy array"
    
    if Sqs.ndim == 1:
        Sqs = Sqs[None, :]                                  # sometimes, for a single voxel, Sqs is a 1D array, we need to make it 2D

    if f0.ndim == 1:
        f0 = f0[None, :]                                    # the same reason as above, we need to make it 2D

    assert qs.ndim == 2 and qs.shape[1] == 2, "qs must be of shape (M, 2)"

    if isinstance(sigma, (int, float)) or (isinstance(sigma, np.ndarray) and sigma.ndim == 0):
        sigma = sigma * np.ones(Sqs.shape[0] * qs.shape[0])
    if isinstance(sigma, np.ndarray):
        if not (sigma.ndim == 1 and sigma.size == Sqs.shape[0] * qs.shape[0]):
            raise ValueError("Size of sigma is not correct")

    assert thetas.ndim == 2 and thetas.shape[1] == 2, "thetas must be of shape (N, 2)"
    assert weights.ndim == 1 and weights.shape[0] == thetas.shape[0], "weights must match the number of thetas"
    assert Sqs.shape[1] == qs.shape[0], "Sqs must match the number of qs"
    assert f0.shape[0] == Sqs.shape[0] and f0.shape[1] == thetas.shape[0], "f0 shape mismatch"

    if f_hat is not None:
        assert isinstance(f_hat, np.ndarray), "f_hat must be a numpy array"
        if f_hat.ndim == 1:
            f_hat = f_hat[None, :]                          # the same reason as above, we need to make it 2D
        assert f_hat.shape[0] == Sqs.shape[0] and f_hat.shape[1] == thetas.shape[0], "f_hat shape mismatch"
    
    if not (R_csr.shape[0] == R_csr.shape[1] and R_csr.shape[0] == Sqs.shape[0] * qs.shape[0]):
        raise ValueError("Size of R_csr is not correct.")

    return qs, thetas, weights, Sqs, sigma, R_csr, f0, f_hat

def loss_function(qs, thetas, weights, Sqs, R_csr, f_hat, sigma, f0):
    """
    This function computes the original loss function for the D2 multiple voxels model WITH noise and WITH roughness.
    It returns the three terms of the loss function.
    """
    qs, thetas, weights, Sqs, sigma, R_csr, f0, f_hat = data_check(qs, thetas, weights, Sqs, sigma, R_csr, f0, f_hat)

    V = Sqs.shape[0]                                                            # number of voxels
    M = qs.shape[0]                                                             # number of qs
    N = thetas.shape[0]                                                         # number of thetas

    ker_mat = kernel(qs, thetas)                                                # in shape (M, N)

    first_term = np.sum( compare_KL(f_hat, f0, weights) )                       # a scalar                                              

    Sqs_hat = get_Sqs(ker_mat, weights, f_hat)                                  # in shape (V, M)

    rSqs = Sqs.ravel(order='C')                                                 # in shape (V*M, )
    rSqs_hat = Sqs_hat.ravel(order='C')                                         # in shape (V*M, )
    side_vector = rSqs - rSqs_hat                                               # in shape (V*M, )
    second_term = 0.5 * np.linalg.norm(side_vector / sigma)**2                  # a scalar

    third_term = 0.5 * rSqs_hat @ R_csr @ rSqs_hat                              # a scalar
    return first_term, second_term, third_term                                  # three scalars

def f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize):

    """
    This function computes the pdf hat{f}_thetas given the dual variables Lambdas.

    Lambdas:        in shape (V, M)
    ker_mat:        in shape (M, N)
    weights:        in shape (N, )
    f0:             in shape (V, N)
    """

    V = Lambdas.shape[0]                                                        # number of voxels
    N = ker_mat.shape[1]                                                        # number of thetas

    f_hat = f0 * np.exp( np.clip( - Lambdas @ ker_mat - 1, -700, 700) )         # in shape (V, N)

    if normalize:
        f_hat /= (f_hat @ weights)[:, None]                                     # normalize the pdf
    return f_hat

def negative_dual_function(Lambdas, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize, rtol, atol, maxiter):
    """
    This function defines the objective function to be optimized.
    Attention, this function needs to be MINIMIZED to optimize Lambdas!

    Lambdas:        in shape (V, M)
    ker_mat:  in shape (M, N)
    weights:        in shape (N, )
    Sqs:            in shape (V, M)
    sigma:          a scalar or a vector. the standard deviation of the noise
    R:              in shape (V*M, V*M). The roughness regularization parameter
    f0:             in shape (V, N)
    """
    V = Sqs.shape[0]        # number of voxels
    M = ker_mat.shape[0]    # number of qs
    N = ker_mat.shape[1]    # number of thetas
    
    rSqs = Sqs.ravel(order='C')             # in shape (V*M, )
    rLambdas = Lambdas.ravel(order='C')     # in shape (V*M, )

    nodes = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize = False)                      # in shape (V, N)
    if normalize:
        first_term = np.sum(np.log(nodes @ weights))                                            # a scalar
    else:
        first_term = np.sum(nodes @ weights)                                                    # a scalar

    if (R_csr.nnz == 0):
        second_term = np.dot(rSqs, rLambdas) + 0.5 * np.linalg.norm(rLambdas * sigma)**2        # a scalar          
    else:                                                         
        Sigma_inv_diag = 1 / (sigma**2) * np.ones(V*M)
        middle_matrix = sp.diags(Sigma_inv_diag, format='csr') + R_csr                          # in shape (V*M, V*M)
        side_vector = Sigma_inv_diag * rSqs + rLambdas                                          # in shape (V*M, )
        second_term = 0.5*side_vector @ PETSc_solver(middle_matrix, side_vector, rtol, atol, maxiter) # a scalar

    return first_term + second_term


def admm(   qs, thetas, weights, Sqs, sigma, R_csr, f0 = None, normalize = True,
            Lambdas = None, rho = 1.0, rho_ratio = 3,  dynamic_rho = False,
            beta = 0.5, c = 1e-4, tol = 1e-6, epsilon = 1e-8, maxiter = 100, 
            cg_rtol = 1e-10, cg_atol = 1e-50, cg_maxiter = 1000, 
            admm_tol = 1e-8, admm_maxiter = 100): 
    
    """
    rho_ratio: should be strictly greater than 1
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
            first_term = np.log(nodes @ weights)                                            # in shape (V, )
        else:
            first_term = nodes @ weights                                                    # in shape (V, )
                        
        second_term = 0.5 * rho * np.linalg.norm(Lambdas - zs + us, axis=1)**2              # in shape (V, )

        return first_term + second_term                                                     # in shape (V, )
    
    def dist_grad_hess(Lambdas, zs, us, rho, ker_mat, ker_prod, weights, f0, normalize):
        """
        Here the dist means distributed.
        Lambdas : in shape (V, M)
        zs      : in shape (V, M)
        us      : in shape (V, M)
        """
        M = ker_mat.shape[0]

        f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize=normalize)        # in shape (V, N)
        f_weights = f_hat * weights                                                     # in shape (V, N)
        grad = - f_weights @ ker_mat.T + rho * (Lambdas - zs + us)                      # in shape (V, M)

        hess = np.tensordot(f_weights, ker_prod, axes=([1], [2]))                       # in shape (V, M, M) vn, ijn -> vij
        I = np.eye(M).reshape(1, M, M)                                                  # in shape (V, M, M)
        hess = hess + rho * I                                                           # in shape (V, M, M)

        if normalize:
            tmp = f_weights @ ker_mat.T                                                 # in shape (V, M)
            hess = hess - tmp.reshape(-1, M, 1) @ tmp.reshape(-1, 1, M)                 # in shape (V, M, M)

        return grad, hess
    
    def dist_Newton_Armijo( Lambdas, zs, us, rho, ker_mat, ker_prod, weights, f0, normalize,
                            beta, c, tol, epsilon, maxiter):
        """
        Here the dist means distributed.
        Lambdas : in shape (V, M)
        zs      : in shape (V, M)
        us      : in shape (V, M)
        """

        V = Lambdas.shape[0]

        obj_current = dist_obj(Lambdas, zs, us, rho, ker_mat, weights, f0, normalize)       # in shape (V, )
        active = np.ones(V, dtype=bool)                                                     # in shape (V, )

        loop = 0
        while True:
            grad, hess = dist_grad_hess(Lambdas, zs, us, rho, ker_mat, ker_prod, weights, f0, normalize)

            L = np.linalg.cholesky(hess)                                                    # in shape (V, M, M)
            y = np.linalg.solve(L, grad[..., np.newaxis])                                   # in shape (V, M, 1)
            direction = -np.linalg.solve(L.transpose(0, 2, 1), y).squeeze(-1)               # in shape (V, M)

            dot_product = np.sum(grad * direction, axis=1)                                  # in shape (V, )

            converged = ( -dot_product <= tol )
            active = active & (~converged)
            if not np.any(active):
                break

            step = np.ones(V, dtype=Lambdas.dtype)                                          # in shape (V, )
            searching = active.copy()

            while True:
                if not np.any(searching):
                    break

                candidate = Lambdas + step.reshape(V, 1) * direction                        # in shape (V, M)
                obj_new = dist_obj(candidate, zs, us, rho, ker_mat, weights, f0, normalize) # in shape (V, )

                satisfied = obj_new <= (obj_current + c * step * dot_product)

                searching = searching & (~satisfied)

                step = np.where(searching, step * beta, step)

                give_up = (step < epsilon) & searching
                if np.any(give_up):
                    searching = searching & (~give_up)

            Lambdas = Lambdas + step.reshape(V, 1) * direction

            obj_current = dist_obj(Lambdas, zs, us, rho, ker_mat, weights, f0, normalize)

            loop += 1
            if loop >= maxiter:
                break

        return Lambdas

    
    """
    start -------------------------------------------------------------------------------------------------------------
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
        Lambdas = np.ones((V, M))

    zs = np.ones(V*M)
    us = np.ones(V*M)


    Sigma_inv = sp.diags(1 / (sigma**2), format='csr')


    loop = 0

    obj_history = []
    obj_history.append( negative_dual_function(Lambdas, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize, cg_rtol, cg_atol, cg_maxiter) 
                        + rho/2*np.linalg.norm(Lambdas.ravel(order='C') - zs + us)**2 - rho/2 * np.linalg.norm(us)**2 )

    primal_history = []
    dual_history = []
    rho_history = []

    while True:

        """
        update Lambdas ------------------------------------------------------------------------------------------------
        """

        Lambdas = dist_Newton_Armijo(   Lambdas, zs.reshape(V, M), us.reshape(V, M), rho, ker_mat, ker_prod, weights, f0, normalize,
                                        beta, c, tol, epsilon, maxiter)
            
        obj_history.append( negative_dual_function(Lambdas, ker_mat, weights, Sqs, sigma, R_csr, f0, normalize, cg_rtol, cg_atol, cg_maxiter)
                            + rho/2*np.linalg.norm(Lambdas.ravel(order='C') - zs + us)**2 - rho/2 * np.linalg.norm(us)**2 )

        """
        update zs -----------------------------------------------------------------------------------------------------
        """
        
        mat_A = sp.eye(V * M, format='csr') + rho*(Sigma_inv + R_csr)
        vec_y = rho*(Sigma_inv + R_csr) @ (Lambdas.ravel(order='C') + us) - Sigma_inv @ Sqs.ravel(order='C')
        zs_new = PETSc_solver(mat_A, vec_y, cg_rtol, cg_atol, cg_maxiter)

        """
        check ---------------------------------------------------------------------------------------------------------
        """
        loop += 1

        primal = np.linalg.norm(Lambdas.ravel(order='C') - zs_new)
        dual   = np.linalg.norm(zs_new - zs)

        primal_history.append(primal)
        dual_history.append(dual)

        if primal <= admm_tol and dual <= admm_tol:
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

        us = us + Lambdas.ravel(order='C') - zs

        """
        end -----------------------------------------------------------------------------------------------------------
        """
        
    f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize)  
    history = [obj_history, primal_history, dual_history, rho_history]
    return Lambdas, f_hat, history                                  

def uncertainty(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize):

    qs, thetas, weights, Sqs, sigma, R, f0, _ = data_check(qs, thetas, weights, Sqs, sigma, R, f0)

    V = Sqs.shape[0]
    M = Sqs.shape[1]
    N = thetas.shape[0]

    ker_mat = kernel(qs, thetas)                                                                                    # in shape (M, N)
    ker_prod = np.einsum('ik,jk->ijk', ker_mat, ker_mat)                                                            # in shape (M, M, N)
    f_hat = f_thetas_hat(Lambdas, ker_mat, weights, f0, normalize = normalize)                                      # in shape (V, N)
    Gamma = np.zeros((V*M, V*M))                                                                                    # in shape (V*M, V*M)
    for i in range(V):
        Gamma[i*M:(i+1)*M, i*M:(i+1)*M] = np.einsum('ijn,n->ij', ker_prod, f_hat[i, :] * weights)                   # hess in shape (V*M, V*M)
    if normalize:
        grad = np.einsum('vn,mn->vmn', f_hat, ker_mat)                                                              # in shape (V, M, N)
        grad = np.einsum('vmn,n->vm', grad, weights)                                                                # in shape (V, M)
        Gamma = Gamma - grad.reshape((-1, 1), order='C') @ grad.reshape((1, -1), order='C')                         # in shape (V*M, V*M)
    
    Sigma_inv = np.eye(V*M) * 1 / (sigma**2)                                                                        # in shape (V*M, V*M)
    Sigma = np.eye(V*M) * (sigma**2)                                                                                # in shape (V*M, V*M)

    precision = Gamma @ (Sigma_inv + 2 * R + R @ Sigma @ R) @ Gamma + 2 * Gamma @ (np.eye(V*M) + R @ Sigma) + Sigma # in shape (V*M, V*M)

    # # ---------------------------------------------------------------------------------------------
    # # From the important truth that R = 0 !

    # part = inverse_spd(Gamma + Sigma)

    # fir_der = - part

    # # ---------------------------------------------------------------------------------------------

    # ker_prod2 = np.einsum('ik,jk,mk->ijmk', ker_mat, ker_mat, ker_mat)                                              # in shape (M, M, M, N)
    # Third_D = np.zeros((V*M, V*M, V*M))                                                                             # in shape (V*M, V*M, V*M)
    # for i in range(V):
    #     Third_D[i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M] = np.einsum('ijkn,n->ijk', ker_prod2, f_hat[i, :] * weights) # in shape (V*M, V*M, V*M)
           
    # sec_der = np.einsum('ijk, ia, jb, kc -> abc', Third_D, part, part, part)

    # # ---------------------------------------------------------------------------------------------

    # ker_prod3 = np.einsum('ik,jk,mk,lk->ijmlk', ker_mat, ker_mat, ker_mat, ker_mat)                                 # in shape (M, M, M, N)
    # Fourth_D = np.zeros((V*M, V*M, V*M, V*M))
    # for i in range(V):
    #     Fourth_D[i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M] = np.einsum('ijkln,n->ijkl', ker_prod3, f_hat[i, :] * weights)
    
    # thi_der = (     np.einsum('ijkl, ia, jb, kc, ld -> abcd', Fourth_D, part, part, part, part)
    #             -   3 * np.einsum('ai, ijk, jbc, kd -> abcd', part, Third_D, sec_der, part)     )
    
    # # ---------------------------------------------------------------------------------------------

    return precision    #, fir_der, sec_der, thi_der

