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
        self.prev_players = None
        self.prev_obs = None
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

        obs = utils.featurize(self.agent_names, alive_agents, raw_obs)
        rewards = self._rewards(alive_agents, raw_obs.players, obs, actions)

        dones = {}
        infos = {}

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                infos[self.agent_names[i]] = {
                    "energy": raw_obs.players[i]["energy"]
                }

                if raw_obs.players[i]["status"] != constants.Status.STATUS_PLAYING.value:
                    infos[self.agent_names[i]]["gold"] = self.prev_players[i]["score"]
                    infos[self.agent_names[i]]["death"] = constants.Status(raw_obs.players[i]["status"])

                    dones[self.agent_names[i]] = True
                    self.count_done += 1

        dones["__all__"] = self.count_done == 4

        return obs, rewards, dones, infos

    def _rewards(self, alive_agents, players, obs, actions):
        rewards = {}

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                sign = 1
                rewards[agent_name] = 0
                # if players[i]["status"] == constants.Status.STATUS_PLAYING.value:
                #     rewards[agent_name] += (players[i]["score"] - self.prev_players[i]["score"]) \
                #                            * constants.SCALE / constants.MAX_GOLD
                if players[i]["status"] != constants.Status.STATUS_STOP_END_STEP.value \
                        and players[i]["status"] != constants.Status.STATUS_PLAYING.value:
                    rewards[agent_name] += -1 - (len(alive_agents) - 1) * 0.1
                    continue

                # if actions[i] == constants.Action.ACTION_CRAFT.value \
                #         and self.prev_obs[agent_name]["conv_features"][12][players[i]["posy"], players[i]["posx"]] == 0:
                #     rewards[agent_name] -= constants.BASE_REWARD * constants.SCALE
                #
                # if actions[i] == constants.Action.ACTION_FREE.value and players[i]["energy"] >= 40:
                #     rewards[agent_name] -= constants.BASE_REWARD * constants.SCALE

                if players[i]["lastAction"] in [0, 1, 2, 3, 5] \
                        and obs[agent_name]["conv_features"][12][players[i]["posy"], players[i]["posx"]]:
                    sign = -1

                if players[i]["lastAction"] == 4:
                    continue

                rewards[agent_name] += sign * (players[i]["energy"] - self.prev_players[i]["energy"]) \
                                       / constants.BASE_ENERGY * constants.BASE_REWARD * constants.SCALE

                # if actions[i] in [0, 1, 2, 3]:
                #     rewards[agent_name] += 0.001

        self.prev_players = players

        return rewards

    def reset(self):
        raw_obs = self.env.reset()
        self.prev_alive = self.agent_names.copy()

        self.prev_players = raw_obs.players.copy()

        self.count_done = 0

        self.prev_obs = utils.featurize(self.agent_names, self.agent_names, raw_obs)

        return self.prev_obs
