import lfi
import torch
import jax
import numpy as np

import lfi.inference.sbi
# import numpy as np
# import sbi
# import sbi.analysis
# import timeit
# import torch
#
# import lfi.utils


# set seeds
np.random.seed(21355)
torch.manual_seed(21)

prior = lfi.priors.UniformPrior(low=-10, high=10, dim=2)
sim = lfi.simulators.BimodalGaussian(sigma_noise=0.1)

prior_sbi = prior.return_sbi_object()

