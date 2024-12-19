import numpy as np

class BaseObservation:
    def __init__(self, name: str):
        self.name = name

    def sample(self, *args, **kwargs):
        raise NotImplementedError


class Zeros(BaseObservation):
    def __init__(self):
        super().__init__(name="zeros")

    def sample(self, nof_obs: int = 1, dim_y: int = 1):
        return np.zeros((nof_obs, dim_y))


class FromList(BaseObservation):
    def __init__(self):
        super().__init__(name="from_list")

    def sample(self, values: list):
        return np.array([values])