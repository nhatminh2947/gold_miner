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

            gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / 3000

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
                            fill_value=obs.players[i]["energy"] / (constants.MAX_ENERGY / 2))
                ]),
                "fc_features": position
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

    gold_next_to_prob = 0.1
    obstacle_prob = [0.75]

    dx = [-1, 0, 0, 1]
    dy = [0, -1, 1, 0]
    n_gold_spots = np.random.randint(15, 25)
    n_digging_times = np.random.randint(100, 200) - n_gold_spots

    map = np.zeros((9, 21), dtype=int)

    while n_gold_spots:
        i = np.random.randint(9)
        j = np.random.randint(21)

        while map[i, j] != 0:
            i = np.random.randint(9)
            j = np.random.randint(21)
        gold_q.append((i, j))
        n_digging_this_spot = 1 + (np.ceil(np.random.normal(10, 5)) if n_gold_spots != 1 else n_digging_times)

        map[i, j] = n_digging_this_spot * 50
        n_digging_times = max(0, n_digging_times - n_digging_this_spot)
        n_gold_spots -= 1

        for ix, iy in zip(dx, dy):
            ii = i + ix
            jj = j + iy

            if 0 <= ii < 9 and 0 <= jj < 21 and np.random.random() < gold_next_to_prob:
                n_digging_this_spot = 1 + (np.ceil(np.random.normal(10, 5)) if n_gold_spots != 1 else n_digging_times)
                map[ii, jj] = n_digging_this_spot * 50
                n_digging_times = max(0, n_digging_times - n_digging_this_spot)
                n_gold_spots -= 1
                gold_q.append((ii, jj))

        obstacle_type = np.random.randint(1, 4)
        while len(gold_q) != 0:
            x, y = gold_q.pop(0)
            for ix, iy in zip(dx, dy):
                xx = x + ix
                yy = y + iy

                if map[xx, yy] == 0 and np.random.random() < 0.75:
                    map[xx, yy] = -obstacle_type

    print(map)
    with open("resources/Maps/map0_0", "w") as f:
        import json
        json.dump(map.tolist(), f)

    return map


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
    generate_map()
