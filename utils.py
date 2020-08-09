import numpy as np

import constants


def policy_mapping(agent_id):
    # agent_id pattern training/opponent_policy-id_agent-num
    # print("Calling to policy mapping {}".format(agent_id))

    return agent_id


def featurize(agent_names, alive_agents, obs):
    land = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    tree = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    trap = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    swamp = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    gold = np.zeros([obs.mapInfo.height + 1, obs.mapInfo.width + 1], dtype=int)
    players = np.zeros((4, obs.mapInfo.height + 1, obs.mapInfo.width + 1), dtype=int)

    for i in range(obs.mapInfo.height + 1):
        for j in range(obs.mapInfo.width + 1):
            obstacle = obs.mapInfo.get_obstacle(j, i)
            land[i, j] = obstacle == constants.Obstacle.LAND.value
            tree[i, j] = obstacle == constants.Obstacle.TREE.value
            trap[i, j] = obstacle == constants.Obstacle.TRAP.value
            swamp[i, j] = obstacle == constants.Obstacle.SWAMP.value
            gold[i, j] = obs.mapInfo.gold_amount(j, i) / constants.MAX_GOLD

    for i in range(4):
        if obs.players[i]["status"] == constants.Status.STATUS_PLAYING.value:
            players[i][obs.players[i]["posy"], obs.players[i]["posx"]] = 1

    board = np.stack([land, tree, trap, swamp, gold])
    board = np.concatenate([board, players])

    featurized_obs = {}

    for i, agent_name in enumerate(agent_names):
        if agent_name in alive_agents:
            featurized_obs[agent_name] = {
                "conv_features": np.copy(board),
                "energy": [obs.players[i]["energy"] / constants.MAX_ENERGY]
            }

            if i + 5 + 1 < 9:
                board[[5, i + 5 + 1]] = board[[i + 5 + 1, 5]]

    return featurized_obs
