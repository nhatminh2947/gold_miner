import numpy as np

import constants


def policy_mapping(agent_id):
    # agent_id pattern training/opponent_policy-id_agent-num
    # print("Calling to policy mapping {}".format(agent_id))

    return agent_id


def featurize(agent_names, alive_agents, obs):
    obstacle_1 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    obstacle_random = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    obstacle_5 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    obstacle_10 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    obstacle_40 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    obstacle_100 = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    gold = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    gold_amount = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    players = np.zeros((4, obs.mapInfo.height + 1, obs.mapInfo.width + 1), dtype=int)

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

            gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / constants.MAX_GOLD

    for i in range(4):
        if obs.players[i]["status"] == constants.Status.STATUS_PLAYING.value:
            players[i][obs.players[i]["posy"], obs.players[i]["posx"]] = 1

    board = np.stack([obstacle_random, obstacle_1, obstacle_5, obstacle_10, obstacle_40, obstacle_100, gold, gold_amount])
    board = np.concatenate([players, board])

    featurized_obs = {}

    for i, agent_name in enumerate(agent_names):
        if agent_name in alive_agents:
            featurized_obs[agent_name] = {
                "conv_features": np.copy(board),
                "energy": [max(0, obs.players[i]["energy"]) / constants.MAX_ENERGY]
            }

        if i + 1 < 9:
            board[[0, i + 1]] = board[[i + 1, 0]]

    return featurized_obs
