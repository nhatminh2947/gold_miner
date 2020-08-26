import ray


@ray.remote(num_cpus=0.1, num_gpus=0)
class Helper:
    def __init__(self, policy_names):
        self.policy_names = policy_names

        self.hyperparams = {
            policy_name: {
                "lr": 1e-4,
                "clip_param": 0.2,
                "entropy_coeff": 0.01,
                "exploration_reward_coeff": 0.01,
                "game_reward_coeff": 0,
                "energy_reward_coeff": 0
            } for policy_name in self.policy_names
        }

    def get_alpha(self, policy_name):
        return self.population[policy_name].alpha

    def get_reward_coeff(self, policy_name):
        return self.hyperparams[policy_name]["exploration_reward_coeff"], \
               self.hyperparams[policy_name]["game_reward_coeff"], \
               self.hyperparams[policy_name]["energy_reward_coeff"]

    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

    def get_hyperparams(self):
        return self.hyperparams
