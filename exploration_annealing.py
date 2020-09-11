import numpy as np
import ray


@ray.remote(num_cpus=0.1, num_gpus=0)
class ExplorationAnnealing:
    def __init__(self, policies, max_timesteps=5e10):
        self.alphas = {policy: 1 for policy in policies}
        self._max_timesteps = max_timesteps

    def update_alpha(self, policy, gold):
        self.alphas[policy] = 1 - np.tanh(0.5 * (gold / 500))

    def get_alpha(self, policy):
        return self.alphas[policy]
