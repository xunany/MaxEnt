import numpy as np
from scipy.stats import multivariate_normal, norm

from Basic_functions import Cartesian, kernel, get_Sqs

def synthetic_f_line(theta1, theta2, voxels = 3, normalize = True):
    """
    This function generates multiple synthetic 2D pdfs with two components, each of which is a bivariate Gaussian distribution.
    It assumes that the multiple pdfs correspond to the a LINE of voxels and thus the pdfs have some patterns.

    theta1:       in shape (n1, ), the first dimension of the 2D theta. Here I assume that it is from 0 to 2
    theta2:       in shape (n2, ), the second dimension of the 2D theta. Here I assume that it is from 0 to 100
    """

    gau1_x = np.linspace(0.3, 1.7, voxels)
    gau1_y = 90 - 40 * np.sin(np.linspace(0.3, np.pi - 0.3, voxels))
    gau1_sigma_x = 0.15 - 0.15*(np.linspace(0.2, 1.7, voxels) - 1)**2
    gau1_sigma_y = 8 - 8*(np.linspace(0.2, 1.7, voxels) - 1)**2
    gau1_rho = np.linspace(-0.5, 0.5, voxels)

    gau2_x = np.linspace(1.6, 0.4, voxels)
    gau2_y = np.sin(np.linspace(0.3, np.pi - 0.3, voxels)) * 30 + 10
    gau2_sigma_x = 0.05 + 0.1*(np.linspace(0.2, 1.7, voxels) - 1)**2
    gau2_sigma_y = 5 + 5*(np.linspace(0.2, 1.7, voxels) - 1)**2
    gau2_rho = np.linspace(0.5, 0, voxels)
    
    # all the voxels has bivariate Gaussian distribution

    thetas, weights = Cartesian(theta1, theta2)

    f_line = np.empty((voxels, thetas.shape[0])) # return a matrix of size (V, N), V is the number of voxels, 

    for i in range(len(gau1_x)):
        mean1 = [gau1_x[i], gau1_y[i]]
        cov1 = [
            [gau1_sigma_x[i]**2, gau1_rho[i]*gau1_sigma_x[i]*gau1_sigma_y[i]],
            [gau1_rho[i]*gau1_sigma_x[i]*gau1_sigma_y[i], gau1_sigma_y[i]**2]
        ]
        comp1 = multivariate_normal(mean=mean1, cov=cov1)

        mean2 = [gau2_x[i], gau2_y[i]]
        cov2 = [
            [gau2_sigma_x[i]**2, gau2_rho[i]*gau2_sigma_x[i]*gau2_sigma_y[i]],
            [gau2_rho[i]*gau2_sigma_x[i]*gau2_sigma_y[i], gau2_sigma_y[i]**2]
        ]
        comp2 = multivariate_normal(mean=mean2, cov=cov2)

        f_thetas = np.zeros(thetas.shape[0])
        for comp, weight in zip([comp1, comp2], [0.5, 0.5]):
            f_thetas += weight * comp.pdf(thetas)

        if normalize:
            f_thetas /= np.sum(f_thetas * weights)  # normalize the pdf

        f_line[i, :] = f_thetas
    
    return thetas, f_line

def synthetic_f_line_uni(theta1, theta2, voxels = 3, normalize = True):
    """
    This function generates multiple synthetic 2D pdfs with two components, each of which is a univariate Gaussian distribution.
    It assumes that the multiple pdfs correspond to the a LINE of voxels and thus the pdfs have some patterns.

    theta1:       in shape (n1, ), the first dimension of the 2D theta. Here I assume that it is from 0 to 2
    theta2:       in shape (n2, ), the second dimension of the 2D theta. Here I assume that it is from 0 to 100
    """

    gau_x = np.linspace(0.3, 1.7, voxels)
    gau_y = 100 - 80 * np.sin(np.linspace(0.3, np.pi - 0.3, voxels))
    gau_sigma_x = 0.2 - 0.15*(np.linspace(0.2, 1.7, voxels) - 1)**2
    gau_sigma_y = 8 - 8*(np.linspace(0.2, 1.7, voxels) - 1)**2
    gau_rho = np.linspace(-0.9, 0.9, voxels)
    
    # all the voxels has bivariate Gaussian distribution

    thetas, weights = Cartesian(theta1, theta2)

    f_line = np.empty((voxels, thetas.shape[0])) # return a matrix of size (V, N), V is the number of voxels, 

    for i in range(len(gau_x)):
        mean = [gau_x[i], gau_y[i]]
        cov = [
            [gau_sigma_x[i]**2, gau_rho[i]*gau_sigma_x[i]*gau_sigma_y[i]],
            [gau_rho[i]*gau_sigma_x[i]*gau_sigma_y[i], gau_sigma_y[i]**2]
        ]
        comp = multivariate_normal(mean=mean, cov=cov)

        f_thetas = comp.pdf(thetas)

        if normalize:
            f_thetas /= np.sum(f_thetas * weights)  # normalize the pdf

        f_line[i, :] = f_thetas
    
    return thetas, f_line

def synthetic_Sqs_line(q1, q2, theta1, theta2, f_line, sigma = 0):
    ## This function generates a synthetic Sqs given the designed qs
    qs, _ = Cartesian(q1, q2)
    thetas, weights = Cartesian(theta1, theta2)
    kernel_matrix = kernel(qs, thetas)
    Sqs_line = np.empty(( f_line.shape[0] , qs.shape[0])) 
    for i in range(f_line.shape[0]):
        Sqs_line[i,] = get_Sqs(kernel_matrix, weights, f_line[i,:], sigma)

    return qs, Sqs_line


def synthetic_f_plane(theta1, theta2, v1 = 9, v2 = 9, normalize = True):

    thetas, weights = Cartesian(theta1, theta2)

    f_plane = np.empty((v1, v2, thetas.shape[0]))  

    for i in range(v1):
        for j in range(v2):
            if i <= 2 or i >= 6:
                if j <= 2 or j >= 6:
                    mean = [1.0, 50]
                    cov = [[0.3**2, 0], [0, 15**2]]
                    comp = multivariate_normal(mean=mean, cov=cov)
                    f_thetas = comp.pdf(thetas)
                else:
                    mean = [1.5, 20]
                    cov = [[0.1**2, 0], [0, 5**2]]
                    comp = multivariate_normal(mean=mean, cov=cov)
                    f_thetas = comp.pdf(thetas)
            else:
                if j <= 2 or j >= 6:
                    mean = [0.5, 80]
                    cov = [[0.05**2, 0], [0, 3**2]]
                    comp = multivariate_normal(mean=mean, cov=cov)
                    f_thetas = comp.pdf(thetas)
                else:
                    mean = [1.5, 20]
                    cov = [[0.1**2, 0], [0, 5**2]]
                    comp = multivariate_normal(mean=mean, cov=cov)
                    f_thetas = comp.pdf(thetas) * 0.5

                    mean = [0.5, 80]
                    cov = [[0.05**2, 0], [0, 3**2]]
                    comp = multivariate_normal(mean=mean, cov=cov)
                    f_thetas = f_thetas + comp.pdf(thetas) * 0.5
        
            if normalize:
                f_thetas /= np.sum(f_thetas * weights)  # normalize the pdf
            f_plane[i, j, :] = f_thetas

    return thetas, f_plane

def synthetic_Sqs_plane(q1, q2, theta1, theta2, f_plane, sigma = 0):

    qs, _ = Cartesian(q1, q2)
    thetas, weights = Cartesian(theta1, theta2)
    kernel_matrix = kernel(qs, thetas)
    Sqs_plane = np.empty(( f_plane.shape[0], f_plane.shape[1] , qs.shape[0])) 
    for i in range(f_plane.shape[0]):
        for j in range(f_plane.shape[1]):
            Sqs_plane[i, j,] = get_Sqs(kernel_matrix, weights, f_plane[i, j, :], sigma)

    return qs, Sqs_plane

def synthetic_f_plane_1D(thetas, v1 = 9, v2 = 9, normalize = True):
    
    weights = np.gradient(thetas.ravel())

    f_plane = np.empty((v1, v2, thetas.shape[0]))  

    for i in range(v1):
        for j in range(v2):
            if i <= 2 or i >= 6:
                if j <= 2 or j >= 6:
                    mean = 1.0
                    std = 0.25
                    comp = norm(loc=mean, scale=std)
                    f_thetas = comp.pdf(thetas)
                else:
                    mean = 1.5
                    std = 0.1
                    comp = norm(loc=mean, scale=std)
                    f_thetas = comp.pdf(thetas)
            else:
                if j <= 2 or j >= 6:
                    mean = 0.5
                    std = 0.1
                    comp = norm(loc=mean, scale=std)
                    f_thetas = comp.pdf(thetas)
                else:
                    mean = 1.5
                    std = 0.1
                    comp = norm(loc=mean, scale=std)
                    f_thetas = comp.pdf(thetas) * 0.5

                    mean = 0.5
                    std = 0.1
                    comp = norm(loc=mean, scale=std)
                    f_thetas = f_thetas + comp.pdf(thetas) * 0.5
        
            if normalize:
                f_thetas /= np.sum(f_thetas * weights)  # normalize the pdf
            f_plane[i, j, :] = f_thetas

    thetas = thetas.reshape((-1, 1), order = 'C')

    return thetas, f_plane

def synthetic_Sqs_plane_1D(qs, thetas, f_plane, sigma = 0):

    weights = np.gradient(thetas.ravel())
    kernel_matrix = kernel(qs, thetas)
    Sqs_plane = np.empty(( f_plane.shape[0], f_plane.shape[1] , qs.shape[0])) 
    for i in range(f_plane.shape[0]):
        for j in range(f_plane.shape[1]):
            Sqs_plane[i, j,] = get_Sqs(kernel_matrix, weights, f_plane[i, j, :], sigma)

    return qs, Sqs_plane

def synthetic_f_line_1D(thetas, voxels = 3, normalize = True):
    
    weights = np.gradient(thetas.ravel())

    f_line = np.empty((voxels, thetas.shape[0])) # return a matrix of size (V, N), V is the number of voxels, 

    mean1 = np.linspace(0.5, 1.5, voxels)
    std1 = 0.25 - 0.5*(mean1 - 1)**2

    mean2 = np.linspace(1.5, 0.5, voxels)
    std2 = 0.25 - 0.5*(mean2 - 1)**2

    for i in range(voxels):
        comp = norm(loc=mean1[i], scale=std1[i])
        f_thetas = comp.pdf(thetas)*0.4

        comp = norm(loc=mean2[i], scale=std2[i])
        f_thetas = f_thetas + comp.pdf(thetas)*0.6

        if normalize:
            f_thetas /= np.sum(f_thetas * weights)  # normalize the pdf
        f_line[i, :] = f_thetas

    thetas = thetas.reshape((-1, 1), order = 'C')

    return thetas, f_line

def synthetic_Sqs_line_1D(qs, thetas, f_line, sigma = 0):

    weights = np.gradient(thetas.ravel())
    kernel_matrix = kernel(qs, thetas)
    Sqs_plane = np.empty(( f_line.shape[0], qs.shape[0])) 
    for i in range(f_line.shape[0]):
        Sqs_plane[i, ] = get_Sqs(kernel_matrix, weights, f_line[i, :], sigma)

    return qs, Sqs_plane



