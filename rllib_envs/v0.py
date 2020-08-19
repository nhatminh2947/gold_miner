import copy

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import constants
import utils
from MinerTraining import Metrics
from MinerTraining import MinerEnv
from constants import Action


class RllibMinerEnv(MultiAgentEnv):
    def __init__(self, config):
        self.env = MinerEnv(config["host"], config["port"])
        self.env.start()
        self.agent_names = None

        self.policy_names = [
            "policy_0",
            "policy_1",
            "policy_2",
            "policy_3",
        ]

        self.is_render = config["render"]

        self.prev_gold_map = None
        self.prev_raw_obs = None
        self.prev_score = [0, 0, 0, 0]
        self.episode_len = 0

        self.count_done = 0
        self.stat = []
        for i in range(4):
            self.stat.append({metric.name: 0 for metric in Metrics})
            self.stat[i][Metrics.ENERGY.name] = 50
        self.total_gold = 0

    def step(self, action_dict):
        actions = []
        for i in range(4):
            if self.agent_names[i] in action_dict:
                actions.append(action_dict[self.agent_names[i]])
                self.stat[i][Metrics(actions[-1]).name] += 1
            else:
                actions.append(Action.ACTION_FREE.value)

        if self.is_render:
            print(f"Action: {[constants.Action(action).name for action in actions]}")
            utils.print_map(self.prev_raw_obs)

        alive_agents = list(action_dict.keys())
        raw_obs = self.env.step(','.join([str(action) for action in actions]))

        obs = utils.featurize_v2(self.agent_names, alive_agents, raw_obs, self.total_gold)
        rewards, win_loss = self._rewards_v1(alive_agents, raw_obs.players, raw_obs)

        dones = {}
        infos = {}
        self.episode_len += 1

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                infos[self.agent_names[i]] = {}

                self.stat[i][Metrics.ENERGY.name] += raw_obs.players[i]["energy"]

                if raw_obs.players[i]["status"] != constants.Status.STATUS_PLAYING.value:
                    self.stat[i][Metrics.ENERGY.name] /= (self.episode_len + 1)
                    infos[self.agent_names[i]]["win"] = win_loss[self.agent_names[i]]
                    infos[self.agent_names[i]]["gold"] = raw_obs.players[i]["score"]
                    infos[self.agent_names[i]]["status"] = constants.Status(raw_obs.players[i]["status"])
                    infos[self.agent_names[i]]["metrics"] = self.stat[i]
                    dones[self.agent_names[i]] = True
                    self.count_done += 1

        dones["__all__"] = self.count_done == 4
        self.prev_raw_obs = copy.deepcopy(raw_obs)

        if self.is_render:
            print(f"rewards: {rewards}")
            print(f"Energy ", end='')
            for i in range(4):
                print(f"({i}):{raw_obs.players[i]['energy']:5}\t", end='')
            print()

        return obs, rewards, dones, infos

    def _rewards(self, alive_agents, players, raw_obs):
        rewards = {}
        win_loss = {}

        max_score = -1
        max_energy = -1
        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents and players[i]["status"] in [constants.Status.STATUS_STOP_END_STEP.value,
                                                                       constants.Status.STATUS_STOP_EMPTY_GOLD.value]:
                if max_score < players[i]["score"]:
                    max_score = players[i]["score"]
                    max_energy = players[i]["energy"]
                elif max_score == players[i]["score"] and max_energy < players[i]["energy"]:
                    max_energy = players[i]["energy"]

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                rewards[agent_name] = (players[i]["score"] - self.prev_score[i]) * 1.0 \
                                      / constants.MAX_EXTRACTABLE_GOLD

                if players[i]["status"] in [constants.Status.STATUS_STOP_END_STEP.value,
                                            constants.Status.STATUS_STOP_EMPTY_GOLD.value]:
                    if players[i]["score"] == 0:
                        rewards[agent_name] = -1
                        win_loss[agent_name] = 0
                    elif players[i]["score"] == max_score:
                        if players[i]["energy"] >= max_energy:
                            rewards[agent_name] = 1
                            win_loss[agent_name] = 1
                        else:
                            rewards[agent_name] = -1
                            win_loss[agent_name] = 0
                    else:
                        rewards[agent_name] = -1
                        win_loss[agent_name] = 0
                elif players[i]["status"] in [constants.Status.STATUS_ELIMINATED_WENT_OUT_MAP.value,
                                              constants.Status.STATUS_ELIMINATED_OUT_OF_ENERGY.value]:
                    rewards[agent_name] = -1.5
                    win_loss[agent_name] = 0

                if players[i]["lastAction"] == constants.Action.ACTION_CRAFT.value \
                        and self.prev_raw_obs.mapInfo.gold_amount(players[i]["posx"], players[i]["posy"]) == 0:
                    rewards[agent_name] -= 0.01
                elif players[i]["lastAction"] in [constants.Action.ACTION_GO_UP.value,
                                                  constants.Action.ACTION_GO_DOWN.value,
                                                  constants.Action.ACTION_GO_LEFT.value,
                                                  constants.Action.ACTION_GO_RIGHT.value] \
                        and raw_obs.mapInfo.gold_amount(players[i]["posx"], players[i]["posy"]):
                    rewards[agent_name] += 0.001

                self.prev_score[i] = players[i]["score"]

        return rewards, win_loss

    def _rewards_v1(self, alive_agents, players, raw_obs):
        rewards = {}
        win_loss = {}

        max_score = -1
        max_energy = -1
        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents and players[i]["status"] in [constants.Status.STATUS_STOP_END_STEP.value,
                                                                       constants.Status.STATUS_STOP_EMPTY_GOLD.value]:
                if max_score < players[i]["score"]:
                    max_score = players[i]["score"]
                    max_energy = players[i]["energy"]
                elif max_score == players[i]["score"] and max_energy < players[i]["energy"]:
                    max_energy = players[i]["energy"]

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                rewards[agent_name] = (players[i]["score"] - self.prev_score[i]) * 1.0 \
                                      / constants.MAX_EXTRACTABLE_GOLD

                if players[i]["status"] in [constants.Status.STATUS_STOP_END_STEP.value,
                                            constants.Status.STATUS_STOP_EMPTY_GOLD.value]:
                    if players[i]["score"] == 0:
                        rewards[agent_name] = -1
                        win_loss[agent_name] = 0
                    elif players[i]["score"] == max_score:
                        if players[i]["energy"] >= max_energy:
                            # rewards[agent_name] = 1
                            win_loss[agent_name] = 1
                        else:
                            # rewards[agent_name] = -1
                            win_loss[agent_name] = 0
                    else:
                        # rewards[agent_name] = -1
                        win_loss[agent_name] = 0

                elif players[i]["status"] in [constants.Status.STATUS_ELIMINATED_WENT_OUT_MAP.value,
                                              constants.Status.STATUS_ELIMINATED_OUT_OF_ENERGY.value]:
                    rewards[agent_name] = -1
                    win_loss[agent_name] = 0

                if players[i]["lastAction"] == constants.Action.ACTION_CRAFT.value \
                        and self.prev_raw_obs.mapInfo.gold_amount(players[i]["posx"], players[i]["posy"]) == 0:
                    rewards[agent_name] -= 0.01
                    self.stat[i][Metrics.INVALID_CRAFT.name] += 1
                elif players[i]["lastAction"] in [constants.Action.ACTION_GO_UP.value,
                                                  constants.Action.ACTION_GO_DOWN.value,
                                                  constants.Action.ACTION_GO_LEFT.value,
                                                  constants.Action.ACTION_GO_RIGHT.value] \
                        and raw_obs.mapInfo.gold_amount(players[i]["posx"], players[i]["posy"]):
                    rewards[agent_name] += 0.001

                self.prev_score[i] = players[i]["score"]

        return rewards, win_loss

    def reset(self):
        raw_obs = self.env.reset()

        self.total_gold = 0
        for cell in raw_obs.mapInfo.golds:
            self.total_gold += cell["amount"]

        self.prev_score = [0, 0, 0, 0]
        self.count_done = 0
        self.prev_raw_obs = copy.deepcopy(raw_obs)
        self.episode_len = 0

        self.agent_names = list(np.random.permutation(self.policy_names))
        for i in range(4):
            self.agent_names[i] = f"{self.agent_names[i]}_{i}"

        self.stat = []
        for i in range(4):
            self.stat.append({metric.name: 0 for metric in Metrics})
            self.stat[i][Metrics.ENERGY.name] = 50

        return utils.featurize_v2(self.agent_names, self.agent_names, raw_obs, self.total_gold)
