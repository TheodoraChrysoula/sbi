import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
from .base import InferenceBase


# Create Rejection abc class
class ABCRejection(InferenceBase):
    def __init__(
            self,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
            tolerance: float
    ):
        # Compute dimensions from the prior and the observation
        dim = prior.dim
        dim_y = observation.shape[1]

        # Intialize the base class
        super().__init__("ABC Rejection", prior, simulator, observation, dim, dim_y)

        # ABC rejection tolerance
        self.tolerance = tolerance

        self.posterior = None

    def fit(self, budget: int = 1_000):
        # Sample from the prior
        thetas = self.prior.sample_numpy(budget)
        #print(thetas.shape)
        
        # Compute the simulated data 
        sim = self.simulator.sample_numpy(thetas)
        # print(sim.shape)
        # print(f"Print the first 10 simulated data: {sim[:10]}")

        # Compute the distances
        distances = np.linalg.norm(self.observation - sim, axis=1)
        #print(distances.shape)
        #print(f"Print the first 10 distances: {distances[:10]}")

        # Identify accepted samples (satisfy the distance criterion)
        accepted_indices = np.where(distances < self.tolerance)[0]
        best_indices = np.argsort(distances)
        accepted_samples = thetas[best_indices]
        self.posterior = accepted_samples

        
    
    def sample(self, nof_samples: int = 100):
        '''
        Draw samples from the posterior. If fewer samples than requested 
        were accepted, returns all available samples.
        '''

        if self.posterior is None:
            raise ValueError("Posterior is not computed yet.")
        
        # Handle case where fewer than requested samples were accepted
        available_samples = self.posterior.shape[0]
        if nof_samples > available_samples:
            print(f"Only {available_samples} accepted samples available. Returning all")
            samples = self.posterior
            return samples
        else:
            # Sample from the posterior
            samples = self.posterior[:nof_samples]
            return samples
        