import numpy as np
from ray.rllib.utils.schedules import ConstantSchedule


class PopulationBasedTraining:
    def __init__(self, perturb_prob=0.1, perturb_val=0.2, burn_in=5e7, ready=5e7):
        self.perturb_prob = perturb_prob
        self.perturb_val = perturb_val
        self.burn_in = burn_in
        self.ready = ready
        self.last_update = 0

        self.hyperparameters = {"lr": (1e-5, 1e-3),
                                "clip_param": (0.1, 0.3),
                                "entropy_coeff": (0.001, 0.2)}

    def exploit(self, trainer, src, dest):
        self.copy_weight(trainer, src, dest)

    def copy_weight(self, trainer, src, dest):
        P0key_P1val = {}
        for (k, v), (k2, v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                    trainer.get_policy(src).get_weights().items()):
            P0key_P1val[k] = v2

        trainer.set_weights({dest: P0key_P1val,
                             src: trainer.get_policy(src).get_weights()})

        for (k, v), (k2, v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                    trainer.get_policy(src).get_weights().items()):
            assert (v == v2).all()

    def explore(self, trainer, policy_name):
        policy = trainer.get_policy(policy_name)

        new_lr = self.explore_helper(policy.cur_lr, self.hyperparameters["lr"])
        policy.lr_schedule = ConstantSchedule(new_lr, framework="torch")

        new_clip_param = self.explore_helper(policy.config["clip_param"], self.hyperparameters["clip_param"])
        policy.config["clip_param"] = new_clip_param

        new_entropy_coeff = self.explore_helper(policy.config["entropy_coeff"], self.hyperparameters["entropy_coeff"])
        policy.config["entropy_coeff"] = new_entropy_coeff

    def explore_helper(self, old_value, range):
        if np.random.random() > self.perturb_prob:  # resample
            return np.random.uniform(low=range[0], high=range[1], size=None)

        if np.random.random() < 0.5:  # perturb_val = 0.8
            return old_value * (1 - self.perturb_val)

        return old_value * (1 + self.perturb_val)

    def run(self, trainer, result):
        max_average_gold = -1
        min_average_gold = 10000

        weakest_agent = None
        strongest_agent = None

        for i in range(4):
            if result["custom_metrics"][f"policy_{i}/gold_mean"] < min_average_gold:
                min_average_gold = result["custom_metrics"][f"policy_{i}/gold_mean"]
                weakest_agent = i

            if result["custom_metrics"][f"policy_{i}/gold_mean"] > max_average_gold:
                max_average_gold = result["custom_metrics"][f"policy_{i}/gold_mean"]
                strongest_agent = i

        self.exploit(trainer, f"policy_{strongest_agent}", f"policy_{weakest_agent}")
        self.explore(trainer, f"policy_{weakest_agent}")
