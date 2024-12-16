import numpy as np
import torch
import lfi
import sbi

np.random.seed(21355)
torch.manual_seed(21)

D_list = [2] # , 5, 10, 20] # 50, 100]
D_list.reverse()
budget_list = [1_000] # ,5_000 10_000, 20_000, 50_000, 100_000]
budget_list.reverse()

for i, D in enumerate(D_list):
    subset_dims = [i for i in range(D)] if D < 10 else [i for i in range(10)]
    limits = [-10, 10]

    # set up the experiment
    sim = lfi.simulators.BimodalGaussian(sigma_noise=0.1)
    prior = lfi.priors.UniformPrior(low=-10, high=10, dim=D)
    observation = np.zeros((1, D))
    posterior_modes = np.ones((2, D))
    posterior_modes[0, :] = -3.
    posterior_modes[1, :] = 3.

    density_estimator_fun = sbi.neural_nets.posterior_nn(
        model='nsf',
        hidden_features=100,# 20, # 50,
        num_transforms=8,# 2, # 5,
        z_score_x="independent",
        z_score_theta="independent",
    )

    for j, budget in enumerate(budget_list):
        inference = lfi.inference.from_sbi.NPECSingleRound(
            prior=prior,
            simulator=sim,
            observation=observation
        )

        samples, time = inference.fit_and_sample(budget, 100, density_estimator=density_estimator_fun)

        _ = inference.plot_training_summary(
            budget=budget,
            savefig="../reports/figures/bimodal_npe_c/D_%d_budget_%d_training_summary.png" % (D, budget)
        )

        _ = inference.plot_posterior_samples(
            samples=samples,
            budget=budget,
            posterior_modes=posterior_modes,
            subset_dims=subset_dims,
            savefig="../reports/figures/bimodal_npe_c/D_%d_budget_%d_posterior_samples.png" % (D, budget)
        )