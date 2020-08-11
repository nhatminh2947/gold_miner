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

        alive_agents = list(action_dict.keys())
        raw_obs = self.env.step(','.join([str(action) for action in actions]))
        rewards = self._rewards(alive_agents, raw_obs.players)

        dones = {}
        infos = {}

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                infos[self.agent_names[i]] = {
                    "energy": raw_obs.players[i]["energy"]
                }

                if raw_obs.players[i]["status"] != constants.Status.STATUS_PLAYING.value:
                    infos[self.agent_names[i]]["gold"] = self.prev_score[agent_name]
                    dones[self.agent_names[i]] = True
                    self.count_done += 1

        dones["__all__"] = self.count_done == 4

        obs = utils.featurize(self.agent_names, alive_agents, raw_obs)

        return obs, rewards, dones, infos

    def _rewards(self, alive_agents, players):
        rewards = {}

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                if players[i]["status"] == constants.Status.STATUS_PLAYING.value:
                    rewards[agent_name] = 0.01 \
                                          + (players[i]["score"] - self.prev_score[agent_name]) / constants.MAX_GOLD
                elif players[i]["status"] != constants.Status.STATUS_STOP_END_STEP:
                    rewards[agent_name] = -1 - (len(alive_agents) - 1) * 0.1

                self.prev_score[agent_name] = players[i]["score"]

        return rewards

    def reset(self):
        raw_obs = self.env.reset()
        self.prev_alive = self.agent_names.copy()

        self.prev_score = {
            agent_name: 0 for agent_name in self.agent_names
        }

        self.count_done = 0

        return utils.featurize(self.agent_names, self.agent_names, raw_obs)
