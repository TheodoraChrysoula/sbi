import sbibm.metrics
import numpy as np
import torch


def c2st(x: np.ndarray[np.float32], y: np.ndarray[np.float32]) -> float:
    return sbibm.metrics.c2st(torch.tensor(x), torch.tensor(y)).detach().item()