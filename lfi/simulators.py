import numpy as np
import torch
import jax
from jax import random
import jax.numpy as jnp


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

    def simulate_pytorch(self, theta):
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
                    return jnp.dot(theta, x) + jax.random.normal(key) * self.sigma_noise

                return jax.vmap(simulate_one, in_axes=(0, None, 0))(theta, x, keys)

class GaussianNoise:
    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise

    def simulate_numpy(self, theta):
        return np.random.normal(theta, self.sigma_noise)

    def simulate_jax(self, theta, keys):
        def simulate_one(theta, key):
            return theta + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0, 0))(theta, keys)

    def simulate_pytorch(self, theta):
        return theta + torch.randn_like(theta)*self.sigma_noise


class BimodalGaussian:
    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise

    def simulate_numpy(self, theta):
        # for each theta in the batch select either the first or the second mode
        mode = np.random.choice([0, 1], size=theta.shape[0])

        mean = theta + 3
        mean[mode == 1] = theta[mode == 1] - 3
        return np.random.normal(mean, self.sigma_noise)

    def simulate_pytorch(self, theta):
        mode = torch.randint(0, 2, (theta.shape[0],))

        mean = theta + 3
        mean[mode == 1] = theta[mode == 1] - 3
        return mean + torch.randn_like(theta)*self.sigma_noise



