import torch

import sys
sys.path.append("/work/xunan/MaxEnt/Code")
from Basic_functions_torch import get_Sqs_sampling

def D2_dist_para_generator(modes, v):
    """
    This function return a matrix stroing random parameters of 2D multivariate mixture normal distributions.
    However, the feature of the distributions is defined within the function but not from the input of the function except modes.

    modes:  for each distribution, how many modes does it mix
    v:      generate "v" random distributions in total
    
    parameters: 3D array, dimension 1 corresponds to number
                          dimension 2 corresponds to modes
                          dimension 3 corresponds to parametes of each distribution. It has 6 parameters. 
                                      0th - 5th correspond to mu_1, mu_2, std_1, std_2, rho, and weights of the mode.  
    """
    theta1_mu_range = torch.tensor([0.05, 2])
    theta1_std_range = torch.tensor([0.01, 0.15])

    theta2_mu_range = torch.tensor([10, 200])
    theta2_std_range = torch.tensor([1, 20])
    
    parameters = torch.zeros((v, modes, 6))

    left_weights = torch.ones(v)

    for i in range(modes):
        parameters[:, i, 0] = theta1_mu_range[0]  + (theta1_mu_range[1]  - theta1_mu_range[0])  * torch.rand(v)
        parameters[:, i, 1] = theta2_mu_range[0]  + (theta2_mu_range[1]  - theta2_mu_range[0])  * torch.rand(v)
        parameters[:, i, 2] = theta1_std_range[0] + (theta1_std_range[1] - theta1_std_range[0]) * torch.rand(v)
        parameters[:, i, 2] = torch.min(parameters[:, i, 2], parameters[:, i, 0] / 5)  # make sure std1 is not too large
        parameters[:, i, 3] = theta2_std_range[0] + (theta2_std_range[1] - theta2_std_range[0]) * torch.rand(v)
        parameters[:, i, 3] = torch.min(parameters[:, i, 3], parameters[:, i, 1] / 5)  # make sure std2 is not too large
        parameters[:, i, 4] = 2.0 * torch.rand(v) - 1.0   # rho parameter in 2d multinormal, it is in [-1, 1]

        if i < modes - 1:
            w = torch.rand(v) * left_weights
            parameters[:, i, 5] = w
            left_weights = left_weights - w
        else:
            parameters[:, i, 5] = left_weights
    return parameters

def D2_dist_sampler(parameters, s):
    """
    Return samples from parameters.

    samples: in shape(V, S, 2). s is the sample size
    """

    v = parameters.shape[0]
    modes = parameters.shape[1]

    modes_indicator = torch.multinomial(parameters[:, :, 5] , num_samples=s, replacement=True)
    samples = torch.zeros((v, s, 2))
    
    for i in range(modes):
        mu = torch.zeros((v, 2))
        mu[:, 0] = parameters[:, i, 0]
        mu[:, 1] = parameters[:, i, 1]

        L = torch.zeros(v, 2, 2)
        L[:, 0, 0] = parameters[:, i, 2]
        L[:, 1, 0] = parameters[:, i, 3] * parameters[:, i, 4]
        L[:, 1, 1] = parameters[:, i, 3] * torch.sqrt(1 - parameters[:, i, 4]**2)

        dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L )

        temp = dist.sample((s,)).permute(1, 0, 2)
        
        # for any negative values points, replace the values with the mu_s
        temp[:, :, 0][temp[:, :, 0] < 0] = mu[:, None, 0].expand(v, s)[temp[:, :, 0] < 0]
        temp[:, :, 1][temp[:, :, 0] < 0] = mu[:, None, 1].expand(v, s)[temp[:, :, 0] < 0]

        temp[:, :, 0][temp[:, :, 1] < 0] = mu[:, None, 0].expand(v, s)[temp[:, :, 1] < 0]
        temp[:, :, 1][temp[:, :, 1] < 0] = mu[:, None, 1].expand(v, s)[temp[:, :, 1] < 0]
        
        samples[modes_indicator == i, :] = temp[modes_indicator == i, :]
    
    return samples              

def synthetic_Sqs(qs, samples, keeps = 1, SNR = None):
    """
    This function generates synthetic Sqs given the designed qs and distribution parameters

    qs:             in shape(M, 2)
    samples:        in shape(V, S, 2)
    keeps:          in scalar. A large number of sampling points are needed to calculate Sqs,
                    but only a small number of points corresponding to Sqs are needed to train the model
    SNR:            if None, no noise; otherwise, SNR = max(S) / sigma
    """

    Sqs = get_Sqs_sampling(qs, samples, sigma = 0)      # in shape (V, M)

    if SNR is not None :
        noise = torch.randn_like(Sqs) * (torch.max(Sqs, dim = 1).values / SNR).unsqueeze(1)
        Sqs = Sqs + noise

    if samples.shape[1] > keeps:
        samples = samples[:, 0:keeps, :]

    return samples, Sqs