import numpy as np

import constants


def policy_mapping(agent_id):
    # agent_id pattern training/opponent_policy-id_agent-num
    # print("Calling to policy mapping {}".format(agent_id))

    return agent_id


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

            gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / total_gold

    max_extractable_gold = gold_amount.copy()

    count_pos = {}

    for i in range(4):
        if obs.players[i]["status"] == constants.Status.STATUS_PLAYING.value:
            players[i][obs.players[i]["posy"], obs.players[i]["posx"]] = 1
            if (obs.players[i]["posy"], obs.players[i]["posx"]) not in count_pos:
                count_pos[obs.players[i]["posy"], obs.players[i]["posx"]] = 0
            count_pos[obs.players[i]["posy"], obs.players[i]["posx"]] += 1

    for i in count_pos:
        max_extractable_gold[i] = max_extractable_gold[i] / count_pos[i]

    board = np.stack(
        [obstacle_random, obstacle_1, obstacle_5, obstacle_10, obstacle_40, obstacle_100, obstacle_value_min,
         obstacle_value_max, gold, gold_amount, max_extractable_gold])
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

    return featurized_obs
