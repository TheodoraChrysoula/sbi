import sbi
import sbi.inference
import lfi
import torch

low = -2
high = 2
D = 2
sigma_noise = 0.1


prior = lfi.priors.UniformPrior(low, high, D)
sim = lfi.simulators.GaussianNoise(sigma_noise)

prior = prior.sbi_uniform()
inference = sbi.inference.NPE_A(
    prior=prior,
)

theta_star = torch.Tensor([0.5, 0.5])
obs = sim.simulate_pytorch(theta_star)

proposal=prior
for _ in range(2):
    theta = proposal.sample((100,))
    x = sim.simulate_pytorch(theta)
    inference.append_simulations(theta, x, proposal=proposal)
    _ = inference.train()
    posterior = inference.build_posterior().set_default_x(obs)
    proposal = posterior
