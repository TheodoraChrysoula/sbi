import numpy as np
import typing
import timeit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator

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

    def plot_posterior_samples(
            self,
            samples: np.ndarray, # (N, Dy)
            budget: int,
            posterior_modes: np.ndarray = None, # (N, Dy)
            subset_dims: typing.Union[None, list] = None,
            savefig=None
    ):
        # Convert samples to a DataFrame for easier Seaborn handling
        samples_df = pd.DataFrame(
            samples,
            columns=[f"x_{i + 1}" for i in range(samples.shape[1])]
        )

        # Create the pairplot
        g = sns.pairplot(
            data=samples_df,
            vars=[f"x_{i + 1}" for i in subset_dims],
            kind="scatter",  # Base pairplot kind
            diag_kind="kde",  # KDE for diagonal plots
            plot_kws={'alpha': 0.5},  # Scatter plot transparency
        )

        g.fig.suptitle(f"{self.name}: D={self.dim}, budget={budget}")

        # Optionally add posterior mode points
        if posterior_modes is not None:
            posterior_modes_df = pd.DataFrame(posterior_modes, columns=[f"x_{i + 1}" for i in range(samples.shape[1])])
            for dim_x in subset_dims:
                for dim_y in subset_dims:
                    if dim_x != dim_y: # Skip diagonal plots
                        ax = g.axes[subset_dims.index(dim_y), subset_dims.index(dim_x)]
                        ax.scatter(
                            posterior_modes_df[f"x_{dim_x + 1}"],
                            posterior_modes_df[f"x_{dim_y + 1}"],
                            color='red', label='Posterior Modes'
                        )

        # Save figure if a path is provided
        if savefig:
            plt.savefig(savefig)

        # Show the plot
        plt.show()
        return g

