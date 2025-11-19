import numpy as np

import sys
import importlib
from pathlib import Path
sys.path.append(str(Path.home() / "Projects" / "MaxEnt" / "Code"))
import Basic_functions
importlib.reload(Basic_functions)
from Basic_functions import inverse_spd, compare_KL, ridge_seq


def Cartesian(thetas):
    """
    This function computes the Cartesian coordinates of two vectors.
    It's equivalent to generating a meshgrid and then flattening it.
    It also computes the weights of the quadrature.

    thetas: which is a vector of size (n,), and it is in ascending order
    """

    weights = np.gradient(thetas)                                                   # in shape (N,)
    thetas = thetas.reshape((-1, 1), order = 'C')

    return thetas, weights

def kernel(qs, thetas):
    """
    This function computes the kernel matrix for the given qs and thetas.
    """
    return np.exp( - qs @ thetas.T )                                                # in shape (M, N). M = qs.shape[0], N = thetas.shape[0]

def get_Sqs(qs, thetas, weights, f_thetas, sigma = 0):
    """
    This function computes the results of a forward model, that is, the y in y = Ax.

    qs:         in shape (M, 1)
    thetas:     in shape (N, 1)
    weights:    in shape (N, )
    f_thetas:   in shape (N, )
    sigma:      noise level, a scalar, default is 0
    """
    return kernel(qs, thetas) @ (f_thetas * weights) + np.random.normal(size = qs.shape[0], scale = sigma)     # in shape (M,)


def data_check(qs, thetas, weights, Sqs, sigma, R, f0, f_hat = None):
    """
    This function checks the data types and shapes of the input parameters.
    """
    assert isinstance(qs, np.ndarray), "qs must be a numpy array"
    assert isinstance(thetas, np.ndarray), "thetas must be a numpy array"
    assert isinstance(weights, np.ndarray), "weights must be a numpy array"
    assert isinstance(Sqs, np.ndarray), "Sqs must be a numpy array"
    assert isinstance(sigma, (int, float)), "sigma must be a scalar"
    assert isinstance(R, np.ndarray), "R must be a numpy array"
    assert isinstance(f0, np.ndarray), "f0 must be a numpy array"
    
    if Sqs.ndim == 1:
        Sqs = Sqs[None, :]      # sometimes, for a single voxel, Sqs is a 1D array, we need to make it 2D

    if f0.ndim == 1:
        f0 = f0[None, :]        # the same reason as above, we need to make it 2D

    assert qs.ndim == 2 and qs.shape[1] == 1, "qs must be of shape (M, 1)"
    assert thetas.ndim == 2 and thetas.shape[1] == 1, "thetas must be of shape (N, 1)"
    assert weights.ndim == 1 and weights.shape[0] == thetas.shape[0], "weights must match the number of thetas"
    assert Sqs.shape[1] == qs.shape[0], "Sqs must match the number of qs"
    assert f0.shape[0] == Sqs.shape[0] and f0.shape[1] == thetas.shape[0], "f0 shape mismatch"

    if f_hat is not None:
        assert isinstance(f_hat, np.ndarray), "f_hat must be a numpy array"
        if f_hat.ndim == 1:
            f_hat = f_hat[None, :]  # the same reason as above, we need to make it 2D
        assert f_hat.shape[0] == Sqs.shape[0] and f_hat.shape[1] == thetas.shape[0], "f_hat shape mismatch"

    return qs, thetas, weights, Sqs, sigma, R, f0, f_hat

def loss_function(qs, thetas, weights, Sqs, R, f_hat, sigma, f0):
    """
    This function computes the original loss function for the D2 multiple voxels model WITH noise and WITH roughness.
    It returns the three terms of the loss function.
    """
    qs, thetas, weights, Sqs, sigma, R, f0, f_hat = data_check(qs, thetas, weights, Sqs, sigma, R, f0, f_hat)

    V = Sqs.shape[0]                # number of voxels
    M = qs.shape[0]                 # number of qs
    N = thetas.shape[0]             # number of thetas

    first_term = np.sum( compare_KL(f_hat, f0, weights) )                       # a scalar
    
    Sigma_inv_diag = 1 / (sigma**2) * np.ones(V*M)                              # in shape (V*M, V*M)
    Sqs_hat = np.zeros((V, M))                                                  # in shape (V, M)
    for i in range(V):
        Sqs_hat[i, :] = get_Sqs(qs, thetas, weights, f_hat[i, :])
    rSqs = Sqs.ravel(order='C')                                                 # in shape (V*M, )
    rSqs_hat = Sqs_hat.ravel(order='C')                                         # in shape (V*M, )
    side_vector = rSqs - rSqs_hat                                               # in shape (V*M, )
    second_term = 0.5 * side_vector @ np.diag(Sigma_inv_diag) @ side_vector     # a scalar

    third_term = 0.5 * rSqs_hat @ R @ rSqs_hat                                  # a scalar, the roughness regularization term
    return first_term, second_term, third_term

def f_thetas_hat(Lambdas, qs, thetas, weights, f0, normalize):

    """
    This function computes the pdf hat{f}_thetas given the dual variables Lambdas.

    Lambdas:    in shape (V, M)
    qs:         in shape (M, 2)
    thetas:     in shape (N, 2)
    weights:    in shape (N, )
    f0:         in shape (V, N)
    """
    V = Lambdas.shape[0]                # number of voxels
    N = thetas.shape[0]                 # number of thetas

    kernel_matrix = kernel(qs, thetas)  # in shape (M, N)
    f_hat = f0 * np.exp( np.clip( - Lambdas @ kernel_matrix - 1, -700, 700) ) # in shape (V, N)

    if normalize:
        f_hat /= np.tile((f_hat @ weights)[:, None], (1, N))  # normalize the pdf
    return f_hat

def negative_dual_function(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize):
    """
    This function defines the objective function to be optimized.
    Attention, this function needs to be MINIMIZED to optimize Lambdas!

    Lambdas:    in shape (V, M)
    qs:         in shape (M, 2)
    thetas:     in shape (N, 2)
    weights:    in shape (N, )
    Sqs:        in shape (V, M)
    sigma:      a scalar. Which is the noise level, the standard deviation of the noise
    R:          in shape (V*M, V*M). The roughness regularization parameter
    f0:         in shape (V, N)
    """
    V = Lambdas.shape[0]                    # number of voxels
    M = qs.shape[0]                         # number of qs
    N = thetas.shape[0]
    
    rSqs = Sqs.ravel(order='C')             # in shape (V*M, )
    rLambdas = Lambdas.ravel(order='C')     # in shape (V*M, )

    nodes = f_thetas_hat(Lambdas, qs, thetas, weights, f0, normalize = False)                   # in shape (V, N)
    if normalize:
        first_term = np.sum(np.log(nodes @ weights))                                            # a scalar
    else:
        first_term = np.sum(nodes @ weights)                                                    # a scalar

    if np.all(R == 0):
        Sigma_diag = sigma**2 * np.ones(V*M)                                                    # in shape (V*M, )
        second_term = np.dot(rSqs, rLambdas) + 0.5 * rLambdas @ np.diag(Sigma_diag) @ rLambdas  # a scalar          
    else:                                                         
        Sigma_inv_diag = 1 / (sigma**2) * np.ones(V*M)
        middle_matrix = np.diag(Sigma_inv_diag) + R                                             # in shape (V*M, V*M)
        side_vector = Sigma_inv_diag * rSqs + rLambdas                                          # in shape (V*M, )
        second_term = 0.5*side_vector @ inverse_spd(middle_matrix, side_vector, use_cg = False) # a scalar

    return first_term + second_term

def gradient_hessian(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize):
    """
    This function computes the gradient and the hessian matrix for the negative dual function.

    Lambdas:    in shape (V, M)
    qs:         in shape (M, 2)
    thetas:     in shape (N, 2)
    weights:    in shape (N, )
    Sqs:        in shape (V, M)
    f0:         in shape (V, N)
    """
    V = Lambdas.shape[0]                                                                        # number of voxels
    M = qs.shape[0]                                                                             # number of qs
    N = thetas.shape[0]
        
    rSqs = Sqs.ravel(order='C')                                                                 # in shape (V*M, )
    rLambdas = Lambdas.ravel(order='C')                                                         # in shape (V*M, )

    Sigma_inv_diag = 1 / (sigma**2) * np.ones(V*M)                                              # in shape (V*M, )
    middle_matrix = np.diag(Sigma_inv_diag) + R                                                 # in shape (V*M, V*M)
    side_vector = Sigma_inv_diag * rSqs + rLambdas                                              # in shape (V*M, )

    Sigma_diag = (sigma**2) * np.ones(V*M)                                                      # in shape (V*M, )

    ker = kernel(qs, thetas)                                                                    # in shape (M, N)
    f_hat = f_thetas_hat(Lambdas, qs, thetas, weights, f0, normalize = normalize)               # in shape (V, N)
    grad1 = np.einsum('vn,mn->vmn', f_hat, ker)                                                 # in shape (V, M, N)
    grad1 = np.einsum('vmn,n->vm', grad1, weights)                                              # in shape (V, M)
    if np.all(R == 0):
        grad2 = rSqs + np.diag(Sigma_diag) @ rLambdas
    else:
        grad2 = inverse_spd(middle_matrix, side_vector, use_cg=False)                           # in shape (V*M, )
    gradient = - grad1.ravel(order='C') + grad2                                                 # in shape (V*M, )

    ker_product = np.einsum('ik,jk->ijk', ker, ker)                                             # in shape (M, M, N)
    hess = np.zeros((V*M, V*M))                                                                 # in shape (V*M, V*M)
    for i in range(V):
        hess[i*M:(i+1)*M, i*M:(i+1)*M] = np.einsum('ijn,n->ij', ker_product, f_hat[i, :] * weights)             # hess in shape (V*M, V*M)
    if normalize:
        hess = hess - grad1.reshape((-1, 1), order='C') @ grad1.reshape((1, -1), order='C')     # in shape (V*M, V*M)
    if np.all(R == 0):
        hess = hess + np.diag(Sigma_diag)                                                       # in shape (V*M, V*M)
    else:
        hess = hess + inverse_spd(middle_matrix, use_cg=False)                                  # in shape (V*M, V*M)
    return gradient, hess

def Newton_Armijo(  qs, thetas, weights, Sqs, sigma, R, f0 = None, normalize = False,
                    Lambdas = None, 
                    ridge = True, log_max = 0, log_min = -6, n_last = 10, 
                    use_cg = True, pc_method = 'jacobi', cg_rtol = 1e-10, cg_maxiter = 2000, 
                    beta = 0.5, c = 1e-4, tol = 1e-6, epsilon = 1e-8, max_iter = 100):
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
    tol:        tolerance for convergence, default is 1e-6
    epsilon:    the smallest number the step can be, default is 1e-8
    max_iter:   maximum number of iterations, default is 100
    stabilize:  whether to add a small value to the hessian matrix in the first few iterations. Default is True
    """

    V = Sqs.shape[0] if Sqs.ndim == 2 else 1            # number of voxels
    M = qs.shape[0]                                     # number of qs
    N = thetas.shape[0]                                 # number of thetas

    if f0 is None:
        f0 = 1/np.sum(weights)*np.ones((V, N))          # in shape (V, N)

    qs, thetas, weights, Sqs, sigma, R, f0, _ = data_check(qs, thetas, weights, Sqs, sigma, R, f0)

    if Lambdas is None:
        Lambdas = np.ones((V, M))                       # in shape (V, M)

    rLambdas = Lambdas.ravel(order='C')                 # in shape (V*M, )

    if ridge:
        ridges = ridge_seq(max_iter, n_last, log_max, log_min)                                  # generate a sequence of ridge values for stabilization
    else:
        ridges = np.zeros(max_iter)                                                             # no ridge values, just zeros

    loop = 0
    obj_history = []

    while True:
        
        grad, hess = gradient_hessian(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize)  # in shape (V*M, ), (V*M, V*M)
        direction = - inverse_spd(hess, grad, ridges[loop], use_cg, pc_method, cg_rtol, cg_maxiter)

        # Armijo line search
        step = 1.0
        obj_current = negative_dual_function(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize)  
        dot_product = grad @ direction                                                          # in shape (V*M, )

        obj_history.append(obj_current) if loop == 0 else None                                  # append the first objective value

        while True:
            rcandidate = rLambdas + step * direction                                            # in shape (V*M, )
            candidate = rcandidate.reshape(V, M, order='C')                                     # in shape (V, M)
            obj_new = negative_dual_function(candidate, qs, thetas, weights, Sqs, sigma, R, f0, normalize) 

            if obj_new <= obj_current + c * step * dot_product:
                break
            step *= beta
            if step < epsilon:                                                                  # small step size, stop shrinking
                break

        # Update
        rLambdas_new = rLambdas + step * direction
        Lambdas_new = rLambdas_new.reshape(V, M, order='C')                                     # in shape (V, M)
        
        obj_new = negative_dual_function(Lambdas_new, qs, thetas, weights, Sqs, sigma, R, f0, normalize) 
        obj_history.append(obj_new)

        diff = np.linalg.norm(Lambdas_new - Lambdas)                   
        if diff <= tol:
            break

        loop += 1
        if loop >= max_iter:
            print(f"Newton Armijo maximum iterations reached: {max_iter}")
            break

        Lambdas = Lambdas_new                                                                   # in shape (V, M)
        rLambdas = Lambdas.ravel(order='C')                                                     # in shape (V*M, )                                  

    f_hat = f_thetas_hat(Lambdas, qs, thetas, weights, f0, normalize)                           # Update the pdf estimate
    return Lambdas, f_hat, obj_history



def admm(   qs, thetas, weights, Sqs, sigma, R, f0 = None, normalize = False,
            Lambdas = None, rho = 1.0,
            beta = 0.5, c = 1e-4, tol = 1e-6, epsilon = 1e-8, max_iter = 100, 
            ridge = True, log_max = 0, log_min = -6, n_last = 10, 
            use_cg = True, pc_method = 'chol', cg_rtol = 1e-10, cg_maxiter = 2000, 
            admm_tol = 1e-8, admm_max_iter = 100, dynamic_rho = False):
    
    def sub_f_hat(sub_Lambdas, qs, thetas, weights, sub_f0, normalize):
        """
        This function computes the f_hat for A SINGLE voxel.
        Both sub_Lambdas and sub_f0 are 1D arrays.
        """

        kernel_matrix = kernel(qs, thetas)                                                          # in shape (M, N)
        f_hat = sub_f0 * np.exp( np.clip( - sub_Lambdas @ kernel_matrix - 1, -700, 700) )           # in shape (N, )

        if normalize:
            f_hat /= (f_hat @ weights)                                                              # normalize the pdf
        return f_hat                                                

    def sub_obj(sub_Lambdas, sub_zs, sub_us, rho, qs, thetas, weights, sub_f0, normalize):
        """
        This function calculates the objective function value for a single voxel, under ADMM scheme, part 1.
        """

        nodes = sub_f_hat(sub_Lambdas, qs, thetas, weights, sub_f0, normalize = False)              # in shape (N, )
        if normalize:
            first_term = np.sum(np.log(nodes @ weights))                                            # a scalar
        else:
            first_term = np.sum(nodes @ weights)                                                    # a scalar
        
        second_term = 0.5 * rho * np.linalg.norm((sub_Lambdas - sub_zs + sub_us))**2

        return first_term + second_term

    def sub_grad_hess(sub_Lambdas, sub_zs, sub_us, rho, qs, thetas, weights, sub_f0, normalize):
        """
        This function calculates the grad and the hess of the sub objective function value for a single voxel.
        """
        M = qs.shape[0]                                                                             # number of qs
        N = thetas.shape[0]

        ker = kernel(qs, thetas)                                                                    # in shape (M, N)
        f_hat = sub_f_hat(sub_Lambdas, qs, thetas, weights, sub_f0, normalize = normalize)          # in shape (N, )
        grad = - ker @ (f_hat * weights) + rho*(sub_Lambdas - sub_zs + sub_us)                      # in shape (M, )

        ker_product = np.einsum('in,jn->ijn', ker, ker)                                             # in shape (M, M, N)
        hess = np.einsum('ijn,n->ij', ker_product, f_hat * weights) + rho * np.eye(M)               # in shape (M, M)

        if normalize:
            hess -= (ker@(f_hat*weights)).reshape(-1, 1) @ (ker@(f_hat*weights)).reshape(1, -1)     # in shape (M, M)
        return grad, hess

    def sub_Newton_Armijo(  sub_Lambdas, sub_zs, sub_us, rho, qs, thetas, weights, sub_f0, normalize = False,
                            ridge = False, log_max = 0, log_min = -6, n_last = 10, 
                            use_cg = True, pc_method = 'jacobi', cg_rtol = 1e-10, cg_maxiter = 1000, 
                            beta = 0.5, c = 1e-4, tol = 1e-6, epsilon = 1e-8, max_iter = 100):
        """
        This function do optimization for a single voxel.
        """
        if ridge:
            ridges = ridge_seq(max_iter, n_last, log_max, log_min)                                  # generate a sequence of ridge
        else:
            ridges = np.zeros(max_iter)                                                             # no ridge values, just zeros

        loop = 0
        obj_history = []

        while True:
            
            grad, hess = sub_grad_hess(sub_Lambdas, sub_zs, sub_us, rho, qs, thetas, weights, sub_f0, normalize)  # in shape (V*M, ), (V*M, V*M)
            direction = - inverse_spd(hess, grad, ridges[loop], use_cg, pc_method, cg_rtol, cg_maxiter)

            # Armijo line search
            step = 1.0
            obj_current = sub_obj(sub_Lambdas, sub_zs, sub_us, rho, qs, thetas, weights, sub_f0, normalize)  
            dot_product = grad @ direction                                                          # in shape (V*M, )

            obj_history.append(obj_current) if loop == 0 else None                                  # append the first objective value

            while True:
                candidate = sub_Lambdas + step * direction                                          # in shape (V*M, )
                obj_new = sub_obj(candidate, sub_zs, sub_us, rho, qs, thetas, weights, sub_f0, normalize) 

                if obj_new <= obj_current + c * step * dot_product:
                    break
                step *= beta
                if step < epsilon:                                                                  # small step size, stop shrinking
                    break

            # Update
            subLambdas_new = sub_Lambdas + step * direction
            
            obj_new = sub_obj(subLambdas_new, sub_zs, sub_us, rho, qs, thetas, weights, sub_f0, normalize) 
            obj_history.append(obj_new)

            diff = np.linalg.norm(subLambdas_new - sub_Lambdas)                   
            if diff <= tol:
                break

            loop += 1
            if loop >= max_iter:
                print(f"Sub Newton Armijo maximum iterations reached: {max_iter}")
                break

            sub_Lambdas = subLambdas_new
        print(obj_history)                                                                           
        return sub_Lambdas
    
    """
    start -------------------------------------------------------------------------------------------------------------
    """

    V = Sqs.shape[0] if Sqs.ndim == 2 else 1            # number of voxels
    M = qs.shape[0]                                     # number of qs
    N = thetas.shape[0]                                 # number of thetas

    if f0 is None:
        f0 = 1/np.sum(weights)*np.ones((V, N))          # in shape (V, N)

    qs, thetas, weights, Sqs, sigma, R, f0, _ = data_check(qs, thetas, weights, Sqs, sigma, R, f0)

    if Lambdas is None:
        Lambdas = np.ones((V, M))

    zs = np.ones(V*M)
    us = np.ones(V*M)

    Sigma_inv = 1 / (sigma**2) * np.eye(V*M)

    loop = 0

    obj_history = []
    obj_history.append( negative_dual_function(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize) 
                        + rho/2*np.linalg.norm(Lambdas.ravel(order='C') - zs + us)**2   )
    primal_history = []
    dual_history = []

    while True:

        """
        update Lambdas ------------------------------------------------------------------------------------------------
        """

        for i in range(V):
            Lambdas[i] = sub_Newton_Armijo(Lambdas[i],  zs[i*M:(i+1)*M],  us[i*M:(i+1)*M],  rho, 
                                            qs,         thetas,           weights,          f0[i],      normalize,
                                            ridge,      log_max,          log_min,          n_last,
                                            use_cg,     pc_method,        cg_rtol,          cg_maxiter, 
                                            beta,       c,                tol,              epsilon,    max_iter)
        obj_history.append( negative_dual_function(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize)
                            + rho/2*np.linalg.norm(Lambdas.ravel(order='C') - zs + us)**2)

        """
        update zs -----------------------------------------------------------------------------------------------------
        """

        mat_A = np.eye(V*M) + rho*(Sigma_inv + R)
        vec_y = rho*(Sigma_inv + R) @ (Lambdas.ravel(order='C') + us) - Sigma_inv @ Sqs.ravel(order='C')

        zs_new = inverse_spd(mat_A, vec_y, 0, use_cg, pc_method, cg_rtol, cg_maxiter)

        """
        check ---------------------------------------------------------------------------------------------------------
        """
        loop += 1
        print(loop, rho)

        primal = np.linalg.norm(Lambdas.ravel(order='C') - zs)
        dual = rho*np.linalg.norm(zs_new - zs)

        primal_history.append(primal)
        dual_history.append(dual)

        if primal <= admm_tol and dual <= admm_tol:
            break

        if loop >= admm_max_iter:
            print("ADMM maximum iterations reached.")
            break

        zs = zs_new

        """
        modify --------------------------------------------------------------------------------------------------------
        """
        if dynamic_rho:
            if primal > 10 * dual:
                rho = rho * 2
                us = us / 2

            if dual > 10 * primal:
                rho = rho / 2
                us = us * 2

        """
        update zs -----------------------------------------------------------------------------------------------------
        """

        us = us + Lambdas.ravel(order='C') - zs

        """
        end -----------------------------------------------------------------------------------------------------------
        """
        
    f_hat = f_thetas_hat(Lambdas, qs, thetas, weights, f0, normalize)  
    history = [obj_history, primal_history, dual_history]
    return Lambdas, f_hat, history

def uncertainty(Lambdas, qs, thetas, weights, Sqs, sigma, R, f0, normalize):

    qs, thetas, weights, Sqs, sigma, R, f0, _ = data_check(qs, thetas, weights, Sqs, sigma, R, f0)

    V = Sqs.shape[0]
    M = Sqs.shape[1]
    N = thetas.shape[0]

    ker = kernel(qs, thetas)                                                                                        # in shape (M, N)
    ker_product = np.einsum('ik,jk->ijk', ker, ker)                                                                 # in shape (M, M, N)
    f_hat = f_thetas_hat(Lambdas, qs, thetas, weights, f0, normalize = normalize)                                   # in shape (V, N)
    Gamma = np.zeros((V*M, V*M))                                                                                    # in shape (V*M, V*M)
    for i in range(V):
        Gamma[i*M:(i+1)*M, i*M:(i+1)*M] = np.einsum('ijn,n->ij', ker_product, f_hat[i, :] * weights)                # hess in shape (V*M, V*M)
    
    Sigma_inv = np.eye(V*M) * 1 / (sigma**2)                                                                        # in shape (V*M, V*M)
    Sigma = np.eye(V*M) * (sigma**2)                                                                                # in shape (V*M, V*M)

    precision = Gamma @ (Sigma_inv + 2 * R + R @ Sigma @ R) @ Gamma + 2 * Gamma @ (np.eye(V*M) + R @ Sigma) + Sigma # in shape (V*M, V*M)

    # ---------------------------------------------------------------------------------------------
    # From the important truth that R = 0 !

    part = inverse_spd(Gamma + Sigma)

    fir_der = - part

    # ---------------------------------------------------------------------------------------------

    ker_prod2 = np.einsum('ik,jk,mk->ijmk', ker, ker, ker)                                                          # in shape (M, M, M, N)
    Third_D = np.zeros((V*M, V*M, V*M))                                                                             # in shape (V*M, V*M, V*M)
    for i in range(V):
        Third_D[i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M] = np.einsum('ijkn,n->ijk', ker_prod2, f_hat[i, :] * weights) # in shape (V*M, V*M, V*M)
           
    sec_der = np.einsum('ijk, ia, jb, kc -> abc', Third_D, part, part, part)

    # ---------------------------------------------------------------------------------------------

    ker_prod3 = np.einsum('ik,jk,mk,lk->ijmlk', ker, ker, ker, ker)                                                 # in shape (M, M, M, N)
    Fourth_D = np.zeros((V*M, V*M, V*M, V*M))
    for i in range(V):
        Fourth_D[i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M, i*M:(i+1)*M] = np.einsum('ijkln,n->ijkl', ker_prod3, f_hat[i, :] * weights)
    
    thi_der = (     np.einsum('ijkl, ia, jb, kc, ld -> abcd', Fourth_D, part, part, part, part)
                -   3 * np.einsum('ai, ijk, jbc, kd -> abcd', part, Third_D, sec_der, part)     )
    
    # ---------------------------------------------------------------------------------------------

    return precision, fir_der, sec_der, thi_der



