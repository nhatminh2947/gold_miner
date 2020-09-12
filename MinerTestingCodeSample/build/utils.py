import json

import numpy as np

import constants
from color_text import ColorText


def print_map(obs):
    width = 8
    print(f"Steps: {obs.stepCount}")
    print(f"Energy ", end='')
    for i, player in enumerate(obs.players):
        if "energy" in player:
            print(f"({i}):{player['energy']:5}\t", end='')
    print()

    # print(f"Gold   ", end='')
    # for i in range(4):
    #     print(f"({i}):{obs.players[i]['score']:5}\t", end='')
    # print()

    for i in range(constants.N_ROWS):
        for j in range(constants.N_COLS):
            players = ""
            type, value = None, None
            for cell in obs.mapInfo.obstacles:
                if j == cell["posx"] and i == cell["posy"]:
                    type, value = cell["type"], cell["value"]

            text_color = ColorText.CBLACK
            if type is None:
                type = 4


            for k, player in enumerate(obs.players):
                if "status" in player and player["status"] == 0 and j == player["posx"] and i == player["posy"]:
                    if k != 0:
                        players += " "
                    players += str(k)
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
            type, value = None, None
            for cell in obs.mapInfo.obstacles:
                if j == cell["posx"] and i == cell["posy"]:
                    type, value = cell["type"], cell["value"]

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
    print()


def featurize_v3(agent_names, alive_agents, obs, total_gold, prev_actions):
    players = np.zeros((4, obs.mapInfo.height + 1, obs.mapInfo.width + 1), dtype=float)
    obstacle_1 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_random = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_5 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_10 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_20 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_40 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_100 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_value_min = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_value_max = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)

    gold = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    gold_amount = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)

    for i in range(obs.mapInfo.height + 1):
        for j in range(obs.mapInfo.width + 1):
            type, value = obs.mapInfo.get_obstacle_type(j, i)
            if value == 0:  # Forest
                obstacle_random[i, j] = 1
            if value == -1:  # Land
                obstacle_1[i, j] = 1
            if value == -5:  # Swamp 1
                obstacle_5[i, j] = 1
            if value == -10:  # Trap
                obstacle_10[i, j] = 1
            if value == -20:  # Swamp 2
                obstacle_20[i, j] = 1
            if value == -40:  # Swamp 3
                obstacle_40[i, j] = 1
            if value == -100:  # Swamp 4
                obstacle_100[i, j] = 1
            if value is None:  # Gold spot
                gold[i, j] = 1
                value = -4

            obstacle_value_min[i, j] = (-value if value != 0 else 5) / constants.MAX_ENERGY
            obstacle_value_max[i, j] = (-value if value != 0 else 20) / constants.MAX_ENERGY

            gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / 1250

    scores = [0, 0, 0, 0]
    energies = [0, 0, 0, 0]
    for i in range(4):
        if obs.players[i]["status"] == constants.Status.STATUS_PLAYING.value:
            players[i][obs.players[i]["posy"], obs.players[i]["posx"]] = 1
            scores[i] = obs.players[i]["score"] / 50 * 0.02
            energies[i] = max(0, obs.players[i]["energy"]) / (constants.MAX_ENERGY / 2)
        else:
            scores[i] = -1
            energies[i] = max(0, obs.players[i]["energy"]) / (constants.MAX_ENERGY / 2)

    board = np.stack([
        obstacle_random,
        obstacle_1,
        obstacle_5,
        obstacle_10,
        obstacle_20,
        obstacle_40,
        obstacle_100,
        obstacle_value_min,
        obstacle_value_max,
        gold,
        gold_amount
    ])
    # board = np.concatenate([players, board])

    featurized_obs = {}

    for i, agent_name in enumerate(agent_names):
        if agent_name in alive_agents:
            position = np.clip(np.array([obs.players[i]["posy"] / 8 * 2 - 1,
                                         obs.players[i]["posx"] / 20 * 2 - 1]), -1, 1)

            one_hot_last_3_actions = np.zeros((3, 6), dtype=np.float32)
            one_hot_last_3_actions[np.arange(3), prev_actions[i]] = 1
            one_hot_last_3_actions = one_hot_last_3_actions.reshape(-1)

            featurized_obs[agent_name] = {
                "conv_features": np.concatenate([
                    players,
                    np.copy(board),
                    np.full((1, obs.mapInfo.height + 1, obs.mapInfo.width + 1),
                            fill_value=max(0, obs.players[i]["energy"]) / (constants.MAX_ENERGY / 2))
                ]),
                "fc_features": np.concatenate([
                    position,
                    one_hot_last_3_actions,
                    scores,
                    energies
                ])
            }

        if i + 1 < 4:
            players[[0, i + 1]] = players[[i + 1, 0]]
            scores[0], scores[i + 1] = scores[i + 1], scores[0]

    return featurized_obs

def inside_map(row, col):
    return 0 <= row < 9 and 0 <= col < 21
