import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from scripts.bimodal_npe_c import observation


class MixtureDensityNetwork(nn.Module):
    def __init__(self, K, D):
        """
        Mixture Density Network as a nn.Module.

        Args:
            K (int): Number of Gaussian components.
            D (int): Dimensionality of each Gaussian component.
        """
        super(MixtureDensityNetwork, self).__init__()
        self.K = K
        self.D = D

        # Define the network
        self.net = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * D * K),  # Outputs mean and log-sigma for each Gaussian component
        )

    def forward(self, x):
        """
        Forward pass that produces the mixture distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, D).

        Returns:
            torch.distributions.MixtureSameFamily: The mixture distribution.
        """
        phi = self.net(x)
        phi = phi.view(-1, self.K, self.D, 2)  # [batch_size, K, D, 2]

        # Split output into mean and log(sigma)
        mu = phi[..., 0]
        sigma = torch.exp(phi[..., 1])  # Ensure positivity of sigma

        # Create the component distributions (independent Normal distributions)
        component_distributions = dist.Independent(
            dist.Normal(loc=mu, scale=sigma),
            reinterpreted_batch_ndims=1,
        )

        # Create mixture weights (uniform distribution over components)
        mixture_weights = torch.ones(x.size(0), self.K, device=x.device) / self.K
        mixture = dist.MixtureSameFamily(
            mixture_distribution=dist.Categorical(probs=mixture_weights),
            component_distribution=component_distributions,
        )
        return mixture

    def sample(self, N, x_0):
        """
        Sample from the mixture distribution.

        Args:
            N (int): Number of samples.
            x_0 (torch.Tensor): Input tensor (batch_size, D).

        Returns:
            torch.Tensor: Samples of shape (N, D).
        """
        mixture = self.forward(x_0)
        return mixture.sample((N,))

    def logprob(self, th, x_0):
        """
        Compute the log-probability of the samples.

        Args:
            th (torch.Tensor): Samples (batch_size, D).
            x_0 (torch.Tensor): Input tensor (batch_size, D).

        Returns:
            torch.Tensor: Log-probabilities of the samples.
        """
        mixture = self.forward(x_0)
        return mixture.log_prob(th)

    def loss(self, th, x_0):
        """
        Compute the negative log-likelihood loss.

        Args:
            th (torch.Tensor): Samples (batch_size, D).
            x_0 (torch.Tensor): Input tensor (batch_size, D).

        Returns:
            torch.Tensor: Negative log-likelihood loss.
        """
        return -self.logprob(th, x_0).mean()

    def plot_samples(self, samples, kde=False):
        """
        Plots posterior samples.

        Args:
            samples (torch.Tensor): Samples to plot, shape (N, D).
            kde (bool): Whether to use kernel density estimation for plotting.
        """
        samples_np = samples.numpy()  # Convert to NumPy for plotting
        D = samples.shape[1]

        if D != 2 and not kde:
            raise ValueError("Scatter plot is supported only for 2D samples.")
        if kde and D != 1:
            raise ValueError("KDE plot is supported only for 1D samples.")

        if kde:
            sns.kdeplot(samples_np[:, 0], fill=True)
        else:
            plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.7)
            plt.xlabel("x_1")
            plt.ylabel("x_2")
            plt.title("Posterior Samples")
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
        plt.show()


# Example usage
if __name__ == "__main__":
    D = 3  # Dimension of each Gaussian
    K = 2  # Number of Gaussian components

    # Initialize the posterior approximation
    posterior = MixtureDensityNetwork(K, D)
    observation = torch.zeros(1, D)

    # prior
    prior = dist.Uniform(-5 * torch.ones(D), 5 * torch.ones(D))
    th = prior.sample((1000,)).detach()

    def simulator(theta, mixture_weight=0.5):
        N, D = theta.shape
        samples = []

        for i in range(N):
            th = theta[i]
            # Mixture components centered at theta + 3 and theta - 3
            center_pos = th + 3
            center_neg = th - 3

            # Choose which Gaussian to sample from
            choice = np.random.choice([0, 1], p=[mixture_weight, 1 - mixture_weight])
            if choice == 0:
                # Sample from Gaussian centered at theta + 3
                sample = np.random.normal(loc=center_pos, scale=1.0, size=D)
            else:
                # Sample from Gaussian centered at theta - 3
                sample = np.random.normal(loc=center_neg, scale=1.0, size=D)

            samples.append(sample)

        return torch.Tensor(np.array(samples)).detach()
    x = simulator(th.numpy())

    log_prob = posterior.logprob(th, x_0=x)
    loss = posterior.loss(th, x_0=x)
    print("Log probability:", log_prob)

    # train
    optimizer = torch.optim.Adam(posterior.net.parameters(), lr=1e-3)
    loss_list = []
    for i in range(500):
        optimizer.zero_grad()
        loss = posterior.loss(th, x_0=x)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}")


    approx_post_samples = posterior.sample(100, x_0=observation)
    posterior.plot_samples(approx_post_samples[:, 0, :2])

    posterior.net(observation).view(-1, K, D, 2)[0, :, :, 1]