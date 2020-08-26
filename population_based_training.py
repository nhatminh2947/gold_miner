import numpy as np
import ray


class PopulationBasedTraining:
    def __init__(self, policy_names, perturb_prob=0.2, perturb_val=0.2, burn_in=5e7, ready=5e7):
        self.perturb_prob = perturb_prob
        self.perturb_val = perturb_val
        self.burn_in = burn_in
        self.ready = ready

        self.policy_names = policy_names

        self.hyperparams_range = {
            "lr": (1e-4, 1e-3),
            "clip_param": (0.1, 0.3),
            "entropy_coeff": (1e-3, 1e-1),
            "exploration_reward_coeff": (1e-2, 5e-2),
            "game_reward_coeff": (0, 1),
            "energy_reward_coeff": (1e-4, 1e-2)
        }

        self.last_update = {
            policy_name: 0 for policy_name in self.policy_names
        }

    def exploit(self, trainer, src, dest):
        self.copy_weight(trainer, src, dest)

    def copy_weight(self, trainer, src, dest):
        trainer.get_policy(dest).set_state(trainer.get_policy(src).get_state())

    def explore(self, trainer, src, dest):
        helper = ray.get_actor("helper")
        hyperparams = ray.get(helper.get_hyperparams.remote())

        self.copy_weight(trainer, src, dest)

        for param in hyperparams[src]:
            hyperparams[dest][param] = self.explore_helper(hyperparams[src][param],
                                                           self.hyperparams_range[param])

        trainer.workers.local_worker().for_policy(lambda p: p.update_lr_schedule(hyperparams[dest]["lr"]), dest)
        trainer.workers.local_worker().for_policy(lambda p: p.update_clip_param(hyperparams[dest]["clip_param"]), dest)
        trainer.workers.local_worker().for_policy(lambda p: p.update_entropy(hyperparams[dest]["entropy_coeff"]), dest)

        helper.set_hyperparams.remote(hyperparams)

    def explore_helper(self, old_value, range):
        if np.random.random() > self.perturb_prob:  # resample
            return np.random.uniform(low=range[0], high=range[1], size=None)

        if np.random.random() < 0.5:  # perturb_val = 0.8
            return old_value * (1 - self.perturb_val)

        return old_value * (1 + self.perturb_val)

    def is_eligible(self, policy, timesteps):
        return timesteps - self.last_update[policy] > self.ready

    def run(self, trainer, result):
        if self.burn_in >= result["timesteps_total"]:
            return

        min_average_gold = 1000000

        strongest_agents, weakest_agent = [], None
        for policy_name in self.policy_names:
            if self.is_eligible(policy_name, result["timesteps_total"]) \
                    and result["custom_metrics"][f"{policy_name}/gold_mean"] < min_average_gold:
                min_average_gold = result["custom_metrics"][f"{policy_name}/gold_mean"]
                weakest_agent = policy_name

        for policy_name in self.policy_names:
            if min_average_gold / result["custom_metrics"][f"{policy_name}/gold_mean"] < 0.75:
                strongest_agents.append(policy_name)

        # self.exploit(trainer, f"policy_{strongest_agent}", f"policy_{weakest_agent}")
        if strongest_agents and weakest_agent:
            self.last_update[weakest_agent] = result["timesteps_total"]
            the_choosen_one = np.random.choice(strongest_agents)
            self.explore(trainer, the_choosen_one, weakest_agent)
