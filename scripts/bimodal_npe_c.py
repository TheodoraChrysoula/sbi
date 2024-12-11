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

D_list = [2,] # [2, 5]
budget_list = [1_000] #  [1_000, 5_000]

for i, D in enumerate(D_list):
    subset_dims = [i for i in range(D)] if D < 10 else [i for i in range(10)]
    limits = [-10, 10]

    sim = lfi.simulators.BimodalGaussian(sigma_noise=0.1)
    prior = lfi.priors.UniformPrior(low=-10, high=10, dim=D)
    observation = np.zeros(D)
    prior_sbi = prior.sbi_uniform()

    density_estimator_fun = sbi.neural_nets.posterior_nn(
        model='nsf',
        hidden_features=100,# 20, # 50,
        num_transforms=8,# 2, # 5,
        z_score_x="independent",
        z_score_theta="independent",
    )

    for j, budget in enumerate(budget_list):
        npe_c_single_round = lfi.utils.SingleRoundNPEC(
            prior=prior_sbi,
            simulator=sim.simulate_pytorch,
            observation=observation,
            density_estimator=density_estimator_fun,
        )

        tic = timeit.default_timer()
        npe_c_single_round.train(simulation_budget=budget)
        toc = timeit.default_timer()
        print(f"\nTraining time: {toc - tic:.2f} seconds")

        npe_c_single_round.plot_training_summary(budget)

        npe_c_single_round.plot_posterior_samples(subset_dims=subset_dims, limits=limits)
