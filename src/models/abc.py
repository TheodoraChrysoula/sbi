import numpy as np
from simulators import simulator, simulator2

def abc_inference(y, x, theta, e, simulator_func):
    '''
    Perform Approximate Bayesian Computation (ABC) inference.
    
    Parameters:
    y (float): The observed value.
    x (ndarray): The input data
    theta (ndarray): Samples from the prior distribution.
    e (float): The acceptance threshold
    
    Retunrs:
    ndarray: Accepted samples that approximate the posterior distribution.
    '''

    accepted_samples = []
    for i in range(len(theta)):
        sim = simulator_func(theta[i, :], x)
        dis = np.linalg.norm(sim-y)
        if dis < e:
            accepted_samples.append(theta[i,:])
    return np.array(accepted_samples)