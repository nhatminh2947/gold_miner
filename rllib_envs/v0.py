from ray.rllib.env.multi_agent_env import MultiAgentEnv

from Miner_Training_Local_CodeSample import MinerEnv
from constants import Action, Status
import utils

class RllibMinerEnv(MultiAgentEnv):
    def __init__(self, config):
        self.env = MinerEnv(config["host"], config["port"])
        self.env.start()
        self.agent_names = []

    def step(self, action_dict):
        actions = []
        for i in range(4):
            if self.agent_names[i] in action_dict:
                actions.append(action_dict[self.agent_names[i]])
            else:
                actions.append(Action.ACTION_FREE.value)

        raw_obs = self.env.step(','.join([str(action) for action in actions]))
        obs = {}
        reward = {}
        dones = {}
        info = {}

        dones["__all__"] = self.env.state.status != Status.STATUS_PLAYING.value

        for agent_name in self.agent_names:
            obs[agent_name] = utils.featurize(raw_obs)

        return fearized_obs

    def reset(self):
        self.env.reset()
