from ray.rllib.env.multi_agent_env import MultiAgentEnv

from Miner_Training_Local_CodeSample import MinerEnv


class RllibPomme(MultiAgentEnv):
    def __init__(self, config):
        self.env = MinerEnv(config["host"], config["port"])
        self.env.start()

    def step(self, action_dict):


    def reset(self):
        self.env.reset()
