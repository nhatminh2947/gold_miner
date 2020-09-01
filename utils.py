import json

import numpy as np

import constants
from color_text import ColorText


def policy_mapping(agent_id):
    # agent_id pattern training/opponent_policy-id_agent-num
    # print("Calling to policy mapping {}".format(agent_id))
    _, id, position = agent_id.split("_")
    return f"policy_{id}"


def featurize(agent_names, alive_agents, obs, total_gold):
    players = np.zeros((4, obs.mapInfo.height + 1, obs.mapInfo.width + 1), dtype=float)
    obstacle_1 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_random = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_5 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_10 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_40 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_100 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_value_min = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_value_max = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)

    gold = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    gold_amount = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)

    for i in range(obs.mapInfo.height + 1):
        for j in range(obs.mapInfo.width + 1):
            type, value = obs.mapInfo.get_obstacle_type(j, i)
            if value == 0:
                obstacle_random[i, j] = 1
            if value == -1:
                obstacle_1[i, j] = 1
            if value == -5:
                obstacle_5[i, j] = 1
            if value == -10:
                obstacle_10[i, j] = 1
            if value == -40:
                obstacle_40[i, j] = 1
            if value == -100:
                obstacle_100[i, j] = 1
            if value is None:
                gold[i, j] = 1
                value = -4

            obstacle_value_min[i, j] = (-value if value != 0 else 5) / constants.MAX_ENERGY
            obstacle_value_max[i, j] = (-value if value != 0 else 20) / constants.MAX_ENERGY

            gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / constants.MAX_EXTRACTABLE_GOLD

    for i in range(4):
        if obs.players[i]["status"] == constants.Status.STATUS_PLAYING.value:
            players[i][obs.players[i]["posy"], obs.players[i]["posx"]] = 1

    board = np.stack(
        [obstacle_random, obstacle_1, obstacle_5, obstacle_10, obstacle_40, obstacle_100, obstacle_value_min,
         obstacle_value_max, gold, gold_amount])
    board = np.concatenate([players, board])

    featurized_obs = {}

    for i, agent_name in enumerate(agent_names):
        if agent_name in alive_agents:
            energy = np.array([max(0, obs.players[i]["energy"]) / constants.MAX_ENERGY])
            position = np.clip(np.array([obs.players[i]["posy"] / 8 * 2 - 1,
                                         obs.players[i]["posx"] / 20 * 2 - 1]), -1, 1)

            featurized_obs[agent_name] = {
                "conv_features": np.copy(board),
                "fc_features": np.concatenate([energy, position])
            }

        if i + 1 < 9:
            board[[0, i + 1]] = board[[i + 1, 0]]

    return


def featurize_v1(agent_names, alive_agents, obs, total_gold):
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

            gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / total_gold

    for i in range(4):
        if obs.players[i]["status"] == constants.Status.STATUS_PLAYING.value:
            players[i][obs.players[i]["posy"], obs.players[i]["posx"]] = 1

    board = np.stack(
        [obstacle_random,
         obstacle_1,
         obstacle_5,
         obstacle_10,
         obstacle_20,
         obstacle_40,
         obstacle_100,
         obstacle_value_min,
         obstacle_value_max,
         gold,
         gold_amount])
    # board = np.concatenate([players, board])

    featurized_obs = {}
    energy_of_agents = []
    score_of_agents = []
    for i, agent_name in enumerate(agent_names):
        if agent_name in alive_agents:
            energy_of_agents.append(obs.players[i]["energy"] / constants.MAX_ENERGY)
            score_of_agents.append(obs.players[i]["score"] / constants.MAX_EXTRACTABLE_GOLD)
        else:
            energy_of_agents.append(0)
            score_of_agents.append(0)

    for i, agent_name in enumerate(agent_names):
        if agent_name in alive_agents:
            position = np.clip(np.array([obs.players[i]["posy"] / 8 * 2 - 1,
                                         obs.players[i]["posx"] / 20 * 2 - 1]), -1, 1)

            featurized_obs[agent_name] = {
                "conv_features": np.concatenate([players, np.copy(board)]),
                "fc_features": np.concatenate([
                    # energy_of_agents,
                    # score_of_agents,
                    position])
            }

        board[[0, i + 1]] = board[[i + 1, 0]]
        if i + 1 < len(energy_of_agents):
            energy_of_agents[0], energy_of_agents[i + 1] = energy_of_agents[i + 1], energy_of_agents[0]
            score_of_agents[0], score_of_agents[i + 1] = score_of_agents[i + 1], score_of_agents[0]

    return featurized_obs


def featurize_v2(agent_names, alive_agents, obs, total_gold):
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

            gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / 2000

    for i in range(4):
        if obs.players[i]["status"] == constants.Status.STATUS_PLAYING.value:
            players[i][obs.players[i]["posy"], obs.players[i]["posx"]] = 1

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

            featurized_obs[agent_name] = {
                "conv_features": np.concatenate([
                    players,
                    np.copy(board),
                    np.full((1, obs.mapInfo.height + 1, obs.mapInfo.width + 1),
                            fill_value=max(0, obs.players[i]["energy"]) / constants.MAX_ENERGY)
                ]),
                "fc_features": position
            }

        if i + 1 < 4:
            players[[0, i + 1]] = players[[i + 1, 0]]

    return featurized_obs


def featurize_lstm_v3(agent_names, alive_agents, obs, total_gold):
    players = np.zeros((4, obs.mapInfo.height + 1, obs.mapInfo.width + 1), dtype=float)
    obstacle_random = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_1 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_5 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_10 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_20 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_40 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)
    obstacle_100 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=float)

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

            gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / 2000

    for i in range(4):
        if obs.players[i]["status"] == constants.Status.STATUS_PLAYING.value:
            players[i][obs.players[i]["posy"], obs.players[i]["posx"]] = 1

    board = np.stack([
        obstacle_random,
        obstacle_1,
        obstacle_5,
        obstacle_10,
        obstacle_20,
        obstacle_40,
        obstacle_100,
        gold,
        gold_amount
    ])
    # board = np.concatenate([players, board])

    featurized_obs = {}

    energy_of_agents = []
    for i, agent_name in enumerate(agent_names):
        if agent_name in alive_agents:
            energy_of_agents.append(max(0, obs.players[i]["energy"]) / (constants.MAX_ENERGY / 2))
        else:
            energy_of_agents.append(0)

    for i, agent_name in enumerate(agent_names):
        if agent_name in alive_agents:
            tmp_energy = energy_of_agents.copy()

            del tmp_energy[i]

            featurized_obs[agent_name] = {
                "conv_features": np.concatenate([
                    players,
                    np.copy(board),
                    np.full((1, obs.mapInfo.height + 1, obs.mapInfo.width + 1),
                            fill_value=max(0, obs.players[i]["energy"]) / (constants.MAX_ENERGY / 2))
                ]),
                "fc_features": np.concatenate([
                    tmp_energy
                ])
            }

        if i + 1 < 4:
            players[[0, i + 1]] = players[[i + 1, 0]]

    return featurized_obs


def print_map(obs):
    width = 8
    print(f"Steps: {obs.stepCount}")
    print(f"Energy ", end='')
    for i in range(4):
        print(f"({i}):{obs.players[i]['energy']:5}\t", end='')
    print()

    print(f"Gold   ", end='')
    for i in range(4):
        print(f"({i}):{obs.players[i]['score']:5}\t", end='')
    print()

    for i in range(constants.N_ROWS):
        for j in range(constants.N_COLS):
            players = ""
            type, _ = obs.mapInfo.get_obstacle_type(j, i)
            text_color = ColorText.CBLACK
            if type is None:
                type = 4

            for k in range(4):
                if j == obs.players[k]["posx"] and i == obs.players[k]["posy"]:
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
    print()


def generate_map():
    gold_q = []

    gold_next_to_prob = 0.20 / 8
    gold_value = [0,
                  50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
                  1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
                  1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]

    gold_prob = [0.08,
                 0.1, 0.1, 0.0725, 0.0725, 0.0725, 0.0725, 0.0535, 0.0535, 0.0535, 0.0535,
                 0.0275, 0.0275, 0.0275, 0.0275, 0.025, 0.025, 0.015, 0.015, 0.01125, 0.01125,
                 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025,
                 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    # print(sum(gold_prob))
    total_gold = 0
    # dx = [-1, 0, 0, 1]
    # dy = [0, -1, 1, 0]
    n_gold_spots = np.random.randint(17, 25)
    # n_digging_times = np.random.randint(100, 150) - n_gold_spots

    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    map = np.zeros((9, 21), dtype=int)
    n_obstacles = 9 * 21 - n_gold_spots

    while n_gold_spots > 0:
        i = np.random.randint(9)
        j = np.random.randint(21)

        while map[i, j] != 0:
            i = np.random.randint(9)
            j = np.random.randint(21)
        gold_q.append((i, j))

        map[i, j] = np.random.choice(gold_value, p=gold_prob)
        total_gold += map[i, j]
        n_gold_spots -= 1

        for ix, iy in zip(dx, dy):
            ii = i + ix
            jj = j + iy

            if 0 <= ii < 9 and 0 <= jj < 21 and np.random.random() < gold_next_to_prob:
                map[ii, jj] = np.random.choice(gold_value, p=gold_prob)
                total_gold += map[ii, jj]
                n_gold_spots -= 1
                gold_q.append((ii, jj))

        obstacle_type = np.random.randint(1, 4)
        while len(gold_q) != 0:
            x, y = gold_q.pop(0)
            for ix, iy in zip(dx, dy):
                xx = x + ix
                yy = y + iy

                if inside_map(xx, yy) and map[xx, yy] == 0 and np.random.random() < 0.75:
                    map[xx, yy] = -obstacle_type
                    n_obstacles -= 1

    obstacle_prob = np.random.uniform(0.25, 0.65)
    for i in range(9):
        for j in range(21):
            if map[i, j] == 0 and np.random.random() < obstacle_prob:
                map[i, j] = -np.random.randint(1, 4)
    # print(total_gold)

    return json.dumps(map.tolist())


def inside_map(row, col):
    return 0 <= row < 9 and 0 <= col < 21


def flip_map():
    import json
    path = "/home/lucius/working/projects/gold_miner/resources/Maps"
    for i in range(1, 6):
        map_id = f"{path}/map{i}"

        with open(map_id, 'r') as fr:
            map = np.asarray(json.load(fr))

            for j, axis in enumerate([0, 1, 0, 1]):
                map = np.flip(map, axis=axis)

                with open(f"./resources/Maps/map{i}_{j}", "w") as fw:
                    json.dump(map.tolist(), fw)


if __name__ == '__main__':
    total_gold = 0
    for i in range(10000):
        _, gold = generate_map()
        total_gold += gold

    print(total_gold / 10000)
