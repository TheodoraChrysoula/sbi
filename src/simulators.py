import numpy as np


def prior(N,dim):
    '''Generate a prior distribution'''
    return np.random.uniform(-2,2, size=(N, dim)).astype(np.float32)

def simulator(theta, x):
    sigma = np.random.normal(0, 0.01)
    sim = np.dot(theta, x) + sigma
    return sim

def simulator2(theta, x):
    # Compute the mean of the normal distribution
    means = np.dot(theta, x)
    # Assume fixed variance
    var = 0.01
    # Generate samples from the normal distributioin with mean and variance
    sim = np.random.normal(means, np.sqrt(var))
    return sim

def simulator_omc(theta, x, u):
    # Compute the mean of the normal distribution
    mean = np.dot(theta, x)
    return mean + u

