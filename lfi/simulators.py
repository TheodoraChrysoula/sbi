import numpy as np
import jax
from jax import random
import jax.numpy as jnp


class UniformPrior:
    def __init__(self, low, high, dim):
        self.low = low
        self.high = high
        self.dim = dim

    def sample_numpy(self, N):
        return np.random.uniform(self.low, self.high, size=(N, self.dim)).astype(np.float32)

    def sample_jax(self, N, keys):
        def sample_one(key):
            return random.uniform(key, shape=(self.dim,), minval=self.low, maxval=self.high, dtype=jnp.float32)
        return jax.vmap(sample_one)(keys)
    

 
class LinearSimulator:
    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise

    def simulate_numpy(self, theta, x):
        return np.dot(theta, x) + np.random.normal(0, self.sigma_noise)

    def simulate_jax(self, theta, x, keys):
        """
        Simulate data for multiple thetas and keys

        Parameters:
        - theta: ndarray, the parameters, shape (N, 2)
        - x: ndarray, the input data, shape (2,)
        - keys: ndarray, the random keys, shape (N, 2)
        """
        def simulate_one(theta, x, key):
            return jnp.dot(theta, x) + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0, None, 0))(theta, x, keys)

