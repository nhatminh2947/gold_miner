from typing import Dict

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy

import arguments
import constants
from MinerTraining import Metrics
from exploration_annealing import ExplorationAnnealing
from models import TorchRNNModel, SecondModel, ThirdModel, FourthModel, FifthModel, SixthModel, SeventhModel
from rllib_envs import v0
from utils import policy_mapping

parser = arguments.get_parser()
args = parser.parse_args()
params = vars(args)


class MinerCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()

        self.training_policies = [f"policy_{i}" for i in range(8)]

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):

        for (agent_name, policy), v in episode.agent_rewards.items():
            info = episode.last_info_for(agent_name)
            episode.custom_metrics[f"{policy}/gold"] = info["gold"]
            episode.custom_metrics[f"{policy}/win"] = info["win"]
            episode.custom_metrics["total_gold"] = info["total_gold"]

            for key in Metrics:
                episode.custom_metrics[f"{policy}/{key.name}"] = info["metrics"][key.name]

            for status in [constants.Status.STATUS_ELIMINATED_OUT_OF_ENERGY,
                           constants.Status.STATUS_ELIMINATED_WENT_OUT_MAP,
                           constants.Status.STATUS_STOP_END_STEP,
                           constants.Status.STATUS_STOP_EMPTY_GOLD]:
                episode.custom_metrics[f"{policy}/{status.name}"] = int(status.name == info["status"].name)

    def on_train_result(self, trainer, result: dict, **kwargs):
        annealing = ray.get_actor("annealing")
        if result["custom_metrics"]:
            for policy in self.training_policies:
                alpha = ray.get(annealing.update_alpha.remote(policy,
                                                              result["custom_metrics"][f"{policy}/gold_mean"]))
                result["custom_metrics"][f"{policy}/alpha"] = alpha


def register(env_config):
    ModelCatalog.register_custom_model("1st_model", TorchRNNModel)
    ModelCatalog.register_custom_model("2nd_model", SecondModel)
    ModelCatalog.register_custom_model("3rd_model", ThirdModel)
    ModelCatalog.register_custom_model("4th_model", FourthModel)
    ModelCatalog.register_custom_model("5th_model", FifthModel)
    ModelCatalog.register_custom_model("6th_model", SixthModel)
    ModelCatalog.register_custom_model("7th_model", SeventhModel)

    tune.register_env("MinerEnv-v0", lambda x: v0.RllibMinerEnv(env_config))


def initialize():
    env_config = {
        "game_state_file": params["game_state_file"],
        "host": "localhost",
        "port": 1234,
        "evaluate": False,
        "render": params["render"]
    }

    register(env_config)

    # Policy setting
    def gen_policy(gamma):
        config = {
            "model": {
                "max_seq_len": params["max_seq_len"],
                "custom_model": params["custom_model"],
                "custom_model_config": {
                    "in_channels": constants.NUM_FEATURES,
                },
                "no_final_linear": True,
            },
            "gamma": gamma,
            "framework": "torch",
            "explore": params["explore"]
        }
        return PPOTorchPolicy, constants.OBS_SPACE, constants.ACT_SPACE, config

    gammas = [0.96, 0.975, 0.9875, 0.99, 0.992, 0.996, 0.998, 0.999]
    policies = {f"policy_{i}": gen_policy(gamma) for i, gamma in enumerate(gammas)}

    policy_names = list(policies.keys())

    ExplorationAnnealing.options(name="annealing").remote(policy_names)

    print("Training policies:", policies.keys())

    return env_config, policies, policy_names


def training_team():
    env_config, policies, policies_to_train = initialize()

    trainer = PPOTrainer

    trials = tune.run(
        trainer,
        restore=params["restore"],
        resume=params["resume"],
        name=params["name"],
        num_samples=params['num_samples'],
        queue_trials=params["queue_trials"],
        stop={
            "timesteps_total": params["timesteps_total"]
        },
        checkpoint_freq=params["checkpoint_freq"],
        checkpoint_at_end=True,
        verbose=1,
        config={
            "gamma": params["gamma"],
            "lr": params["lr"],
            "entropy_coeff": params["entropy_coeff"],
            "kl_coeff": params["kl_coeff"],  # disable KL
            "batch_mode": "complete_episodes" if params["complete_episodes"] else "truncate_episodes",
            "rollout_fragment_length": params["rollout_fragment_length"],
            "env": params["ray_env"],
            "env_config": env_config,
            "num_workers": params["num_workers"],
            "num_cpus_per_worker": params["num_cpus_per_worker"],
            "num_envs_per_worker": params["num_envs_per_worker"],
            "num_gpus_per_worker": params["num_gpus_per_worker"],
            "num_gpus": params["num_gpus"],
            "train_batch_size": params["train_batch_size"],
            "sgd_minibatch_size": params["sgd_minibatch_size"],
            "clip_param": params["clip_param"],
            "lambda": params["lambda"],
            "num_sgd_iter": params["num_sgd_iter"],
            "vf_share_layers": True,
            "vf_loss_coeff": params["vf_loss_coeff"],
            "vf_clip_param": params["vf_clip_param"],
            "callbacks": MinerCallbacks,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping,
                "policies_to_train": policies_to_train,
            },
            "clip_actions": False,
            "observation_filter": params["filter"],  # should use MeanStdFilter
            "evaluation_num_episodes": params["evaluation_num_episodes"],
            "evaluation_interval": params["evaluation_interval"],
            "metrics_smoothing_episodes": 100,
            "log_level": "ERROR",
            "framework": "torch"
        }
    )


if __name__ == "__main__":
    print(params)

    ray.shutdown()
    ray.init(local_mode=params["local_mode"])

    training_team()
