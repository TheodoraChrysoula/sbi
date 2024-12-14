# Base class for inference methods

class InferenceBase:
    def __init__(self, prior, simulator, observation, D, Dy, *args, **kwargs):
        self.prior = prior # Callable: (N,) -> (N, D)
        self.simulator = simulator # Callable: (N, D) -> (N, Dy)
        self.observation = observation # (1, Dy)

        self.Dy = Dy
        self.D = D

        # state variables
        self.trained = False
        self.posterior = None

    def fit(self, simulation_budget=1_000):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError
