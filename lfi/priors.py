import numpy as np
import torch
import sbi
import sbi.utils
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

    def sample_pytorch(self, N):
        return torch.rand(N, self.dim) * (self.high - self.low) + self.low

    def sample_jax(self, N, keys):
        def sample_one(key):
            return random.uniform(key, shape=(self.dim,), minval=self.low, maxval=self.high, dtype=jnp.float32)
        return jax.vmap(sample_one)(keys)

    def return_sbi_object(self):
        return sbi.utils.BoxUniform(low=self.low*torch.ones(self.dim), high=self.high*torch.ones(self.dim))
