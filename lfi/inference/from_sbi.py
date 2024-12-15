import sbi
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import simulate_for_sbi, NPE_C, FMPE, NPE_A
import matplotlib.pyplot as plt
import torch
import sbi.analysis
import sbi.neural_nets
import numpy as np
import jax.numpy as jnp
import typing
import pandas as pd
import seaborn as sns
from .base import InferenceBase

class NPEBase(InferenceBase):
    def __init__(
            self,
            prior,
            simulator: callable, # Callable: (N, D) -> (N, Dy)
            observation: typing.Union[np.ndarray, jnp.ndarray, torch.Tensor], # (1, Dy)
            density_estimator: callable,
    ):
        self.density_estimator = density_estimator

        # prepare prior and simulator
        prior, num_parameters, prior_returns_numpy = process_prior(prior)
        simulator = process_simulator(simulator, prior, prior_returns_numpy)
        check_sbi_inputs(simulator, prior)
        dim = num_parameters
        Dy = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__(prior, simulator, observation, dim, Dy)

    def train(self, simulation_budget):
        raise NotImplementedError

    def sample(self, num_samples):
        if self.posterior is None:
            raise ValueError("Posterior is not trained yet.")
        return self.posterior.sample((num_samples,), x=self.observation)

    def plot_training_summary(self, budget, savefig=None):
        fig, ax = plt.subplots()
        ax.set_title("NPE-C (single round): D=%d, budget=%d" % (self.D, budget))
        ax.plot(self.inference_method.summary["training_loss"], ".-", label="tr")
        ax.plot(self.inference_method.summary["validation_loss"], ".-", label="val")
        ax.set_xlim(1, 1000)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss = Negative Log Likelihood")
        ax.legend()
        if savefig:
            plt.savefig(savefig)
        plt.show()
        return fig, ax

    def plot_posterior_samples(
            self,
            samples: np.ndarray, # (N, Dy)
            budget: int,
            posterior_modes: np.ndarray = None, # (N, Dy)
            subset_dims=[0, 1, 2],
            limits=[-10, 10],
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

        g.fig.suptitle(f"NPE-C (single round): D={self.D}, budget={budget}", y=1.02)

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

class NPEASingleRound(NPEBase):
    def __init__(self, prior, simulator, observation):
        super().__init__(prior, simulator, observation, None)

    def train(self, simulation_budget, num_components=10):
        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator, self.prior, num_simulations=simulation_budget)

        self.inference_method = NPE_A(self.prior, num_components=num_components)
        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=500,
            max_num_epochs=1000,
            final_round=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(self.observation)
        return self.posterior


class NPECSingleRound(NPEBase):
    def __init__(self, prior, simulator, observation, density_estimator):
        super().__init__(prior, simulator, observation, density_estimator)

    def train(self, simulation_budget):
        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator, self.prior, num_simulations=simulation_budget)

        # train density estimator
        density_estimator_fun = sbi.neural_nets.posterior_nn(
            model='nsf',
            hidden_features=100,  # 20, # 50,
            num_transforms=8,  # 2, # 5,
            z_score_x="independent",
            z_score_theta="independent",
        )

        self.inference_method = NPE_C(self.prior, density_estimator=density_estimator_fun)
        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=500,
            max_num_epochs=1000,
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(self.observation)
        return self.posterior


class FMPESingleRound(NPEBase):
    def __init__(self, prior, simulator, observation, density_estimator):
        super().__init__(prior, simulator, observation, density_estimator)

    def train(self, simulation_budget):
        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator, self.prior, num_simulations=simulation_budget)

        # # train density estimator
        # density_estimator_fun = sbi.neural_nets.posterior_nn(
        #     model='maf',
        #     hidden_features=100,  # 20, # 50,
        #     num_transforms=8,  # 2, # 5,
        #     z_score_x="independent",
        #     z_score_theta="independent",
        # )

        self.inference_method = FMPE(self.prior)
        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=500,
            max_num_epochs=1000,
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(self.observation)
        return self.posterior