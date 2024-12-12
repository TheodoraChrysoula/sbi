import os, sys
import numpy as np
import sbi
import sbi.analysis
import timeit
import torch
sys.path.insert(0, os.path.abspath('..'))
import lfi
import lfi.utils

np.random.seed(21355)
torch.manual_seed(21)

D = 2
budget = 30_000


sim = lfi.simulators.BimodalGaussian(sigma_noise=0.1)
prior = lfi.priors.UniformPrior(low=-10, high=10, dim=D)
observation = np.zeros(D)
prior_sbi = prior.sbi_uniform()


inference_method = lfi.utils.SingleRoundFMPE(
    prior=prior_sbi,
    simulator=sim.simulate_pytorch,
    observation=observation,
    density_estimator=None
)

tic = timeit.default_timer()
inference_method.train(
    simulation_budget=budget,
)
toc = timeit.default_timer()
print(f"\nTraining time: {toc - tic:.2f} seconds")

inference_method.plot_training_summary(
    budget,
    savefig=None
)

subset_dims = [i for i in range(D)] if D < 10 else [i for i in range(10)]
limits = [-10, 10]
fig, ax = inference_method.plot_posterior_samples(
    budget,
    subset_dims=subset_dims,
    limits=limits,
    savefig=None,
)
