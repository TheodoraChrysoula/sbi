import numpy as np
import torch
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
    
    def sample_pytorch(self, theta):
        return torch.matmul(theta, self.x) + torch.randn_like(theta)*self.sigma_noise
    

class BaseSimulator:
    def __init__(self,):
        pass
    
    def sample_numpy(self, theta):
        raise NotImplementedError
    def sample_jax(self, theta, keys):
        raise NotImplementedError
    def sample_pytorch(self, theta):
        raise NotImplementedError
    

class GaussianNoise(BaseSimulator):
    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise
        super().__init__()

    def sample_numpy(self, theta):
        return np.random.normal(theta, self.sigma_noise)
    
    def sample_jax(self, theta, keys):
        def simulate_one(theta, key):
            return theta + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0,0))(theta, keys)

    def sample_pytorch(self, theta):
        return theta + torch.randn_like(theta)*self.sigma_noise


class BimodalGaussian(BaseSimulator):
    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise
        super().__init__()

    def sample_numpy(self, theta):
        # for each theta in the batch select either the first or the second mode
        mode = np.random.choice([0,1], size = theta.shape[0])

        mean = theta + 3
        mean[mode==1] = theta[mode == 1] - 3
        return np.random.normal(mean, self.sigma_noise)
    
    def sample_jax(self, theta, keys):
        def simulate_one(theta, key):
            mode = jax.random.choice(key, shape=(theta.shape[0],), a=jnp.array([0.1]))
            mean = theta + 3
            mean = jax.ops.index_update(mean, jax.ops.index[mode==1], theta[mode==1]-3)
            return mean + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0,0))(theta, keys)
                                         
    def sample_pytorch(self, theta):
        mode = torch.randint(0, 2, (theta.shape[0],))

        mean = theta + 3
        mean[mode==1] = theta[mode == 1] - 3
        return mean + torch.randn_like(theta) * self.sigma_noise
    

    

