import numpy as np
import jax
from jax import random
import jax.numpy as jnp


def prior(N,dim):
    '''Generate a prior distribution'''
    return np.random.uniform(-2,2, size=(N, dim)).astype(np.float32)

def prior_jax(N, dim, key):
    return random.uniform(key, shape=(N,dim), minval=-2, maxval=2, dtype=jnp.float32)


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

def simulator_jax(theta, x, key):
    # Compute the mean
    means = jnp.dot(theta, x)
    sigma = jnp.sqrt(0.01)
    return means + sigma*random.normal(key)

