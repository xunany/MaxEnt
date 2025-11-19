import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import gaussian_kde
from petsc4py import PETSc
import os
import matplotlib.pyplot as plt
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

    theta11, theta22 = np.meshgrid(theta1, theta2)
    thetas = np.vstack([theta11.flatten(order='F'), theta22.flatten(order='F')]).T  # in shape (n1*n2, 2)

    delta1 = np.gradient(theta1)                                                    # in shape (n1,)
    delta2 = np.gradient(theta2)                                                    # in shape (n2,)

    weights = np.outer(delta1, delta2).reshape(-1, order = 'C')                     # in shape (n1*n2, )

    return thetas, weights

def kernel(qs, thetas):
    """
    This function computes the kernel matrix for the given qs and thetas.
    """
    return np.exp( - qs @ thetas.T )                                                # in shape (M, N). M = qs.shape[0], N = thetas.shape[0]

def get_Sqs(ker_mat, weights, f_thetas, sigma = 0):
    """
    This function computes the results of a forward model, that is, the y in y = Ax.

    ker_mat:        in shape (M, N)
    weights:        in shape (N, )
    f_thetas:       in shape (V, N) or shape (N, )
    sigma:          noise level, a scalar, default is 0
    """
    epsilon = np.random.normal(size = ker_mat.shape[0] if f_thetas.ndim == 1 
                else (f_thetas.shape[0], ker_mat.shape[0]), scale = sigma)
    return (f_thetas * weights) @ ker_mat.T  +  epsilon                             # in shape (V, M)

###################################################################################################
#   mainly designed for comparing two pdfs
###################################################################################################

def compare_KL(f_num, f_den, weights):
    """
    This function computes the KL divergence between two pdfs and the orders matter.
    
    f_num:      in shape (N, ) or shape (V, N), the pdf in the numerator in the KL divergence
    f_den:      in shape (N, ) or shape (V, N), the pdf in the denominator in the KL divergence, no zero entries
    weights:    in shape (N, ), the weights of the quadrature
    """
    assert f_num.ndim == f_den.ndim, "Dimensions of f_num and f_den must match"
    return np.sum(  weights * (f_num * np.log( np.maximum(f_num / np.maximum(f_den, 1e-30), 1e-30) ) ), 
                    axis = 0 if f_num.ndim == 1 else 1 )                            # a scalar if 1D or shape (V, ) if 2D

def compare_L2(f_a, f_b, weights = None):
    """
    This function computes the L2 distance between two pdfs and the orders do not matter.
    
    f_a:        in shape (N, ) or shape (V, N), the pdf in the first argument
    f_b:        in shape (N, ) or shape (V, N), the pdf in the second argument
    weights:    in shape (N, ), the weights of the quadrature. If None, it is assumed to be all ones.
    """
    if weights is None:
        weights = np.ones_like(f_a[0])
    assert f_a.ndim == f_b.ndim, "Dimensions of f_a and f_b must match"
    return np.sqrt(np.sum(weights * ((f_a - f_b) ** 2), axis = 0 if f_a.ndim == 1 else 1))  # a scalar if f_a is 1D or shape (V, ) if 2D

#   ###############################################################################################
#   plot recovered pdf, the true pdf and S(q)
#   ###############################################################################################

def f_from_samples(samples, theta1 = None, theta2 = None):
    """
    This function computes the 2D pdf from the samples.
    theta1 and theta2 are the range of the discretized 2D support.
    
    samples:    in shape (n, 2), the samples from the pdf
    theta1:     in shape (n1, ), the first dimension of the 2D theta, default is None
    theta2:     in shape (n2, ), the second dimension of the 2D theta, default is None
    """

    if theta1 is None or theta2 is None:
        theta1 = np.linspace(np.min(samples[:, 0]), np.max(samples[:, 0]), 100)     # in shape (n1, )
        theta2 = np.linspace(np.min(samples[:, 1]), np.max(samples[:, 1]), 100)     # in shape (n2, )

    thetas, _ = Cartesian(theta1, theta2)                                           # in shape (n1*n2, 2)

    theta11, theta22 = np.meshgrid(theta1, theta2)

    xy = np.vstack([samples[:,0], samples[:,1]])
    kde = gaussian_kde(xy)
    Z = kde(np.vstack([theta11.ravel(), theta22.ravel()])).reshape(theta11.shape)   
    Z_reshaped = Z.reshape(thetas.shape[0], order='F')                              # in shape (n1*n2, )
    return thetas, Z_reshaped

def contourf_compare(theta1, theta2, f_hat = None, qs = None, Sqs = None, f_true = None, savepath = None):
    """
    theta1:     in shape (n1, ), the first dimension of the 2D theta.
    theta2:     in shape (n2, ), the second dimension of the 2D theta.
    f_hat:      in shape (n1*n2, ), the recovered pdf. N = n1 * n2
    """

    theta11, theta22 = np.meshgrid(theta1, theta2)

    if f_hat is None:
        if f_true is None:
            if qs is None or Sqs is None:
                raise ValueError("At least one of f_hat, f_true, qs + Sqs must be provided.")
            else:
                figure_width = 5
                subs = 1
        else:
            if qs is None or Sqs is None:
                figure_width = 5
                subs = 1
            else:
                figure_width = 10.5
                subs = 2
    else:
        if f_true is None:
            if qs is None or Sqs is None:
                figure_width = 5
                subs = 1
            else:
                figure_width = 10.5
                subs = 2
        else:
            if qs is None or Sqs is None:
                figure_width = 10.5
                subs = 2
            else:
                figure_width = 16
                subs = 3

    fig, axs = plt.subplots(1, subs, figsize=(figure_width, 4))
    if subs == 1:
        axs = [axs]

    sub_index = 0
    if f_hat is not None:
        f_hat_reshape = f_hat.reshape(theta11.shape, order='F')
        contourplot = axs[sub_index].contourf(theta11, theta22, f_hat_reshape, levels=50, cmap='hot') #  viridis gist_earth
        axs[sub_index].set_xlabel(r'D (${\mu m}^2/ms$)')
        axs[sub_index].set_ylabel(r'R ($s^{-1}$)')
        axs[sub_index].set_title('Recovered')
        plt.colorbar(contourplot, ax=axs[sub_index])
        sub_index += 1

    if f_true is not None:
        contourplot_true = axs[sub_index].contourf(theta11, theta22, f_true.reshape(theta11.shape, order='F'), levels=50, cmap='hot')
        axs[sub_index].set_xlabel(r'D (${\mu m}^2/ms$)')
        axs[sub_index].set_ylabel(r'R ($s^{-1}$)')
        axs[sub_index].set_title('True')
        plt.colorbar(contourplot_true, ax=axs[sub_index])
        sub_index += 1
    
    if qs is not None and Sqs is not None:
        thetas, weights = Cartesian(theta1, theta2)
        ker_mat = kernel(qs, thetas)
        axs[sub_index].plot(Sqs, 'o', markersize = 10, color = 'gray', alpha = 1, label = 'Observed True')
        if f_true is not None:
            axs[sub_index].plot(get_Sqs(ker_mat, weights, f_true), 'x', markersize = 5, markeredgewidth=1, color = 'blue', alpha = 1, label = 'Noisefree True')
        axs[sub_index].plot(get_Sqs(ker_mat, weights, f_hat), '.', markersize = 3, color = 'red', alpha = 1, label='Recovered')
        axs[sub_index].set_ylabel('S(q)')
        axs[sub_index].set_xlabel('q')
        axs[sub_index].legend()

    if savepath is not None:
        plt.savefig(savepath, format='pdf', bbox_inches='tight')
    plt.show()

#   ###############################################################################################
#   inverse or solve a linear system with a symmetric positive definite matrix
#   ###############################################################################################

def chol_solver(A, y = None):
    """
    This function solves the linear system Ax = y using Cholesky factorization.
    If y is None, it computes the inverse of A.
    Here the solver is designed for small dimension spd A.
    The form of A is dense.
    """
    if y is None:                                           # If y is None, we want to compute the inverse of A
        c, lower = cho_factor(A)
        x = cho_solve((c, lower), np.eye(A.shape[0]))
    else:                                                   # If y is provided, we want to solve the linear system Ax = y
        c, lower = cho_factor(A)
        x = cho_solve((c, lower), y)
    return x

def PETSc_solver(A_csr, y, rtol = 1e-10, atol = 1e-50, maxiter = 1000):
    """
    This function solves the linear system Ax = y using cg method from PETSc package.
    Here the solver is designed for huge dimension spd A.
    The form of A is scipy csr sparse matrix.
    """
    A_PETSc = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
    A_PETSc.assemblyBegin(); A_PETSc.assemblyEnd()

    b = PETSc.Vec().createWithArray(y)
    x = b.duplicate()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A_PETSc)
    ksp.setType('cg')
    ksp.getPC().setType('hypre') 
    ksp.setTolerances(rtol, atol, maxiter)
    ksp.setFromOptions()
    ksp.solve(b, x)

    return x.getArray()

#   ###############################################################################################
#   solve Q^{-1/2} z given Q and z
#   ###############################################################################################

def chol_sample(Q, z):
    L = np.linalg.cholesky(Q)
    x = solve_triangular(L.T, z, lower=False)
    return x

#   ###############################################################################################
#   mask the real data
#   ###############################################################################################

def mask_brain(signal, median_radius = 1, numpass = 4, vol_idx = [0], least_size = 300, keep_top = None):
    """
    signal:                 is a 4D numpy array data, the first 3D are physical space, the last one is q-space
    least_size:             drop reagions with size smaller than least_size 
    keep_top:               keep only top 2 or 3 big regions. If None, skip
    """

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
        print("+-----------------------------------------------------------------------------+")
        print("Number of initial valid regions:", nlb)
        print("Sizes of each regions (1st is background):", sizes)

        keep_labels = np.where(sizes >= least_size)[0]          # return a pool to decide which region are kept, like [0, 2]
        keep_labels = keep_labels[keep_labels != 0]             # drop the original False voxels

        if keep_top is not None:                                # keep only first keep_top regions
            keep_labels = np.where(np.isin(sizes, np.sort(sizes[keep_labels])[::-1][0:keep_top]))[0]
        print("+-----------------------------------------------------------------------------+")
        print("Number of kept regions:", len(keep_labels) )
        print("Sizes of kept regions:", sizes[keep_labels])
        print("+-----------------------------------------------------------------------------+")

        mask = np.isin(labels, keep_labels)                     # use the region pool to decide which voxels will be turned off  

    # -----------------------------------------------------
    coords = np.argwhere(mask)
    N = coords.shape[0]
    shape = mask.shape

    # lin2idx: a 3D array storing the order of mask==True voxels in the coordinate system of the smallest cube

    lin2idx = -np.ones(shape, dtype=int)
    lin_inds = np.ravel_multi_index(coords.T, dims=shape, order='C')
    lin2idx_flat = lin2idx.ravel()
    lin2idx_flat[lin_inds] = np.arange(N)
    lin2idx = lin2idx_flat.reshape(shape)                        

    return mask, lin2idx

#   ###############################################################################################
#   plot the results from the masked data
#   ###############################################################################################

def contourf_mask(theta1, theta2, f_hat, lin2idx, axis = 0, slice = 0):
    """
    For a 3D brian, plot one slice. 
    if axis == 0, plot [slice, :, :]
    if axis == 1, plot [:, slice, :]
    if axis == 2, plot [:, :, slice]
    """

    shape = lin2idx.shape
    shape = tuple(np.delete(shape, axis))   
    rows = shape[0]
    cols = shape[1]

    fig, axs = plt.subplots(cols, rows, figsize=(2*rows, 2*cols)) 
    plt.subplots_adjust(wspace=0, hspace=0)
    theta11, theta22 = np.meshgrid(theta1, theta2)

    index = np.array([0, 0, 0])
    index[axis] = slice

    for i in range(rows):
        for j in range(cols):
            
            ax = axs[cols-1-j,rows-1-i]                     # from my experience, this matches the MRview for any axis
            index[np.arange(3) != axis] = np.array([i, j])

            if lin2idx[tuple(index)] == -1:
                ax.axis("off")
                ax.set_visible(False)
                continue
            else:
                ax.contourf(theta11, theta22, 
                            f_hat[lin2idx[tuple(index)], :].reshape(theta11.shape, order='F'), 
                            levels=50, cmap='hot')
                # ax.axhline(y=0, color='white', linestyle='-', linewidth=1.2)
                # ax.axvline(x=0, color='white', linestyle='-', linewidth=0.5)
                ax.axis("off")
    # plt.savefig(os.path.expanduser('~/Desktop/MRI.png'), format='png')
    plt.show()
    

