import numpy as np
import ray


@ray.remote(num_cpus=0.1, num_gpus=0)
class ExplorationAnnealing:
    def __init__(self, policy_names, alpha_coeff):
        self.alpha_coeff = alpha_coeff

    def update_alpha(self, policy_name, enemy_death_mean):
        self.population[policy_name].alpha = 1 - np.tanh(self.alpha_coeff * enemy_death_mean)
        if self.population[policy_name].num_steps >= self.burn_in:
            return 0.0
        return self.population[policy_name].alpha

    def get_alpha(self, policy_name):
        return self.population[policy_name].alpha