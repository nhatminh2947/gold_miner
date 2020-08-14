from ray.rllib.env.multi_agent_env import MultiAgentEnv

import constants
import utils
from MinerTrainingLocalCodeSample import Metrics
from MinerTrainingLocalCodeSample import MinerEnv
from color_text import ColorText
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

        self.is_render = config["render"]
        self.prev_alive = self.agent_names.copy()
        self.prev_players = None
        self.prev_obs = None
        self.prev_raw_obs = None
        self.count_done = 0
        self.stat = []
        for i in range(4):
            self.stat.append({metric.name: 0 for metric in Metrics})
            self.stat[i][Metrics.ENERGY.name] = 50
        self.total_gold = 0

    def print_map(self, obs):
        width = 11
        for i in range(constants.N_ROWS):
            for v in range(2):
                for j in range(constants.N_COLS):
                    players = ""
                    type, _ = obs.mapInfo.get_obstacle_type(j, i)
                    text_color = ColorText.CBLACK
                    if type is None:
                        type = 4

                    for k in range(v * 2, v * 2 + 2):
                        if j == obs.players[k]["posx"] and i == obs.players[k]["posy"]:
                            if k == 1 or k == 3:
                                players += " "
                            players += str(k) + f"[{obs.players[k]['energy']}]"
                            text_color = ColorText.CWHITE2

                    color = ColorText.CWHITEBG

                    if type == 1:
                        color = ColorText.CGREENBG
                    elif type == 2:
                        color = ColorText.CGREYBG
                    elif type == 3:
                        color = ColorText.CBLUEBG
                    elif type == 4:
                        color = ColorText.CYELLOWBG

                    print(f"{color}{text_color}{players:{width}}{ColorText.CEND}", end="")
                print()
            for j in range(constants.N_COLS):
                type, value = obs.mapInfo.get_obstacle_type(j, i)
                text_color = ColorText.CBLACK
                if type is None:
                    value = obs.mapInfo.gold_amount(j, i)
                    type = 4
                elif type != constants.Obstacle.SWAMP.value:
                    value = ""
                color = ColorText.CWHITEBG

                if type == 1:
                    color = ColorText.CGREENBG
                elif type == 2:
                    color = ColorText.CGREYBG
                elif type == 3:
                    color = ColorText.CBLUEBG
                elif type == 4:
                    color = ColorText.CYELLOWBG

                print(f"{color}{text_color}{str(value):{width}}{ColorText.CEND}", end="")
            print()

    def step(self, action_dict):
        actions = []
        for i in range(4):
            if self.agent_names[i] in action_dict:
                actions.append(action_dict[self.agent_names[i]])
                self.stat[i][Metrics(actions[-1]).name] += 1
            else:
                actions.append(Action.ACTION_FREE.value)

        if self.is_render:
            self.print_map(self.prev_raw_obs)

        alive_agents = list(action_dict.keys())
        raw_obs = self.env.step(','.join([str(action) for action in actions]))

        obs = utils.featurize(self.agent_names, alive_agents, raw_obs, self.total_gold)
        rewards = self._rewards(alive_agents, raw_obs.players, obs)

        print(f"action: {[constants.Action(action).name for action in actions]}")
        print(f"rewards: {rewards}")

        dones = {}
        infos = {}

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                infos[self.agent_names[i]] = {}

                self.stat[i][Metrics.ENERGY.name] += raw_obs.players[i]["energy"]

                if raw_obs.players[i]["status"] != constants.Status.STATUS_PLAYING.value:
                    infos[self.agent_names[i]]["gold"] = self.prev_players[i]["score"]
                    infos[self.agent_names[i]]["death"] = constants.Status(raw_obs.players[i]["status"])
                    infos[self.agent_names[i]]["metrics"] = self.stat[i]
                    dones[self.agent_names[i]] = True
                    self.count_done += 1

        dones["__all__"] = self.count_done == 4
        self.prev_raw_obs = raw_obs
        print("alive", list(action_dict.keys()))
        print("dones", dones)
        return obs, rewards, dones, infos

    def _rewards(self, alive_agents, players, obs):
        rewards = {}

        for i, agent_name in enumerate(self.agent_names):
            if agent_name in alive_agents:
                rewards[agent_name] = 0

                base_reward = -50 if players[i]["score"] - self.prev_players[i]["score"] == 0 \
                    else players[i]["score"] - self.prev_players[i]["score"]

                if players[i]["status"] != constants.Status.STATUS_STOP_END_STEP.value \
                        and players[i]["status"] != constants.Status.STATUS_PLAYING.value:
                    rewards[agent_name] += -1
                    continue

                if players[i]["lastAction"] == 4:
                    continue

                rewards[agent_name] += abs(players[i]["energy"] - self.prev_players[i]["energy"]) \
                                       / constants.BASE_ENERGY * (base_reward / self.total_gold)

        self.prev_players = players

        return rewards

    def reset(self):
        raw_obs = self.env.reset()

        self.total_gold = 0
        for cell in raw_obs.mapInfo.golds:
            self.total_gold += cell["amount"]

        self.prev_alive = self.agent_names.copy()
        self.prev_raw_obs = raw_obs
        self.prev_players = raw_obs.players.copy()
        self.count_done = 0
        self.prev_obs = utils.featurize(self.agent_names, self.agent_names, raw_obs, self.total_gold)

        self.stat = []
        for i in range(4):
            self.stat.append({metric.name: 0 for metric in Metrics})
            self.stat[i][Metrics.ENERGY.name] = 50

        return self.prev_obs
