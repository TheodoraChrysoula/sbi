import lfi
import torch
import numpy as np

np.random.seed(21355)
torch.manual_seed(21)

for D in [2, 5, 10]:
    sim = lfi.simulators.BimodalGaussian(sigma_noise=0.1)
    prior = lfi.priors.UniformPrior(low=-10, high=10, dim=D)
    observation = np.zeros((1, D))

    budget = 10_000
    inference_method = lfi.inference.custom.MixtureDensityNetwork(
        prior=prior,
        simulator=sim,
        observation=observation,
    )

    inference_method.fit(budget=budget, nof_components=2, nof_epochs=2_000)
    samples = inference_method.sample(nof_samples=100)

    inference_method.plot_posterior_samples(
        samples=samples,
        budget=budget,
        subset_dims=[d for d in range(D)],
        savefig=None
    )

