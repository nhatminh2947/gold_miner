from ray.rllib.env.multi_agent_env import MultiAgentEnv

import constants
import utils
from MinerTrainingLocalCodeSample import MinerEnv
from constants import Action


class RllibMinerEnv(MultiAgentEnv):
    def __init__(self, config):
        self.env = MinerEnv(config["host"], config["port"])
        self.env.start()
        self.agent_names = [
            "policy_0",
            "policy_1",
            "policy_2",
            "policy_3",
        ]

        self.prev_alive = self.agent_names.copy()

        self.prev_score = {
            agent_name: 0 for agent_name in self.agent_names
        }

        self.count_done = 0

    def step(self, action_dict):
        actions = []
        for i in range(4):
            if self.agent_names[i] in action_dict:
                actions.append(action_dict[self.agent_names[i]])
            else:
                actions.append(Action.ACTION_FREE.value)
        print("ACTIONS: ", action_dict)
        alive_agents = list(action_dict.keys())
        raw_obs = self.env.step(','.join([str(action) for action in actions]))

        dones = {}
        infos = {}

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents and raw_obs.players[i]["status"] != constants.Status.STATUS_PLAYING.value:
                dones[self.agent_names[i]] = True
                self.count_done += 1
                print("player_{}: DONE".format(i))

        dones["__all__"] = self.count_done == 4

        obs = utils.featurize(self.agent_names, alive_agents, raw_obs)
        rewards = self._rewards(alive_agents, raw_obs.players)

        return obs, rewards, dones, infos

    def _rewards(self, alive_agents, players):
        rewards = {}

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                if players[i]["status"] == constants.Status.STATUS_PLAYING.value:
                    rewards[agent_name] = players[i]["score"] - self.prev_score[agent_name]
                    self.prev_score[agent_name] = players[i]["score"]
                else:
                    rewards[agent_name] = -1
        print("REWARD:", rewards)

        return rewards

    def reset(self):
        print("RESET ENV")
        raw_obs = self.env.reset()
        self.prev_alive = self.agent_names.copy()

        self.prev_score = {
            agent_name: 0 for agent_name in self.agent_names
        }

        self.count_done = 0

        return utils.featurize(self.agent_names, self.agent_names, raw_obs)
