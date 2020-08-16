import ray
import torch
from ray.rllib.agents.ppo import PPOTrainer
from models import ThirdModel, SecondModel
from training import initialize
from utils import policy_mapping
import constants

ray.init(num_cpus=4)

env_config, policies, policies_to_train = initialize()

ppo_agent = PPOTrainer(config={
    "num_gpus": 1,
    "env_config": env_config,
    "num_workers": 0,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping,
        "policies_to_train": policies_to_train,
    },
    "observation_filter": "NoFilter",
    "clip_actions": False,
    "framework": "torch"
}, env="MinerEnv-v0")

id = 940
checkpoint_dir = "/home/lucius/ray_results/gold_miner/PPO_MinerEnv-v0_0_2020-08-15_18-49-40u3ctdeo0"
checkpoint = "{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, id, id)

ppo_agent.restore(checkpoint)

mem_size = 0
weights = ppo_agent.get_policy("policy_0").get_weights()
for key in weights:
    parameters = 1
    for value in weights[key].shape:
        parameters *= value

    mem_size += parameters

    weights[key] = torch.tensor(weights[key])

torch.save(weights,
           "/home/lucius/working/projects/gold_miner/MinerTrainingLocalCodeSample/TrainedModels/model_0.pt")

model = SecondModel(constants.OBS_SPACE, constants.ACT_SPACE, 6, {}, "2rd_model", constants.NUM_FEATURES, None)
model.load_state_dict(torch.load("/home/lucius/working/projects/gold_miner/MinerTrainingLocalCodeSample/TrainedModels/model_0.pt"))

