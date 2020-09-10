import ray


@ray.remote(num_cpus=0.1, num_gpus=0)
class ExplorationAnnealing:
    def __init__(self, max_timesteps=5e10):
        self.alpha = 1
        self._max_timesteps = max_timesteps

    def update_alpha(self, timesteps):
        self.alpha = max(0, 1 - timesteps / self._max_timesteps)

    def get_alpha(self):
        return self.alpha
