import numpy as np
import pandas as pd
import typing
import timeit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
import lfi

class InferenceBase:
    def __init__(
            self,
            name: str,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
            dim: int,
            dim_y: int,
        ):
        self.name = name
        self.prior = prior
        self.simulator = simulator
        self.observation = observation # (1, Dy)

        self.dim_y = dim_y
        self.dim = dim

        self.posterior = None

    def fit(self, budget: int = 1_000, *args, **kwargs):
        raise NotImplementedError

    def sample(self, nof_samples: int = 100, *args, **kwargs):
        raise NotImplementedError

    def fit_and_sample(self, budget, num_samples):
        tic = timeit.default_timer()

        self.fit(budget)
        samples = self.sample(num_samples)
        toc = timeit.default_timer()
        print(f"\nTraining/Sampling time: {toc - tic:.2f} seconds")
        return samples, toc - tic

    @staticmethod
    def plot_posterior_samples(
            samples: np.ndarray, # (N, Dy)
            subset_dims: typing.Union[None, list] = None,
            limits: typing.Union[None, list] = None,
            savefig: typing.Union[None, str] = None,
    ):
        g = lfi.visualization.plot_pairwise_posterior(
            samples=samples,
            subset_dims=subset_dims,
            limits=limits,
            savefig=savefig
        )


    @staticmethod
    def store(samples, path):
        samples_df = pd.DataFrame(
            samples,
            columns=[f"x_{i + 1}" for i in range(samples.shape[1])],
        )
        samples_df.to_csv(path, index=False)
