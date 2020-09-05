from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

from GAME_SOCKET import GameSocket  # in testing version, please use GameSocket instead of GameSocketDummy
from MINER_STATE import State
import constants
import numpy as np
import torch

TreeID = 1
TrapID = 2
SwampID = 3


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()

        self.score_pre = self.state.score  # Storing the last score for designing the reward function

    def start(self):  # connect to server
        self.socket.connect()

    def end(self):  # disconnect server
        self.socket.close()

    def send_map_info(self, request):  # tell server which map to run
        self.socket.send(request)

    def reset(self):  # start new game
        try:
            message = self.socket.receive()  # receive game info from server
            print(message)
            self.state.init_state(message)  # init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action):  # step process
        self.socket.send(action)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            # print("New state: ", message)
            self.state.update_state(message)  # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    # def get_state(self):
    #     obs = self.state
    #
    #     player_channel = np.zeros((4, obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1), dtype=float)
    #     obstacle_1 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #     obstacle_random = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #     obstacle_5 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #     obstacle_10 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #     obstacle_40 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #     obstacle_100 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #     obstacle_value_min = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #     obstacle_value_max = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #
    #     gold = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #     gold_amount = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
    #
    #     for i in range(obs.mapInfo.max_y + 1):
    #         for j in range(obs.mapInfo.max_x + 1):
    #             type, value = None, None
    #             for cell in obs.mapInfo.obstacles:
    #                 if j == cell["posx"] and i == cell["posy"]:
    #                     type, value = cell["type"], cell["value"]
    #
    #             if value == 0:
    #                 obstacle_random[i, j] = 1
    #             if value == -1:
    #                 obstacle_1[i, j] = 1
    #             if value == -5:
    #                 obstacle_5[i, j] = 1
    #             if value == -10:
    #                 obstacle_10[i, j] = 1
    #             if value == -40:
    #                 obstacle_40[i, j] = 1
    #             if value == -100:
    #                 obstacle_100[i, j] = 1
    #             if value is None:
    #                 gold[i, j] = 1
    #                 value = -4
    #
    #             obstacle_value_min[i, j] = (-value if value != 0 else 5) / constants.MAX_ENERGY
    #             obstacle_value_max[i, j] = (-value if value != 0 else 20) / constants.MAX_ENERGY
    #
    #             gold_amount[i, j] = obs.mapInfo.gold_amount(j, i) / constants.MAX_EXTRACTABLE_GOLD
    #
    #     player_channel[0][obs.y, obs.x] = 1
    #
    #     id = 1
    #     for player in obs.players:
    #         if "status" in player and player["status"] == constants.Status.STATUS_PLAYING.value:
    #             if player["playerId"] == obs.id:
    #                 continue
    #
    #             player_channel[id][player["posy"], player["posx"]] = 1
    #             id += 1
    #
    #     board = np.stack(
    #         [obstacle_random, obstacle_1, obstacle_5, obstacle_10, obstacle_40, obstacle_100, obstacle_value_min,
    #          obstacle_value_max, gold, gold_amount])
    #     board = np.concatenate([player_channel, board])
    #
    #     energy = torch.tensor([max(0, obs.energy) / constants.MAX_ENERGY], dtype=torch.float)
    #     position = torch.clamp(torch.tensor([obs.y / 8 * 2 - 1,
    #                                          obs.x / 20 * 2 - 1], dtype=torch.float), -1, 1)
    #
    #     featurized_obs = {
    #         "obs": {
    #             "conv_features": torch.unsqueeze(torch.tensor(board, dtype=torch.float), 0),
    #             "fc_features": torch.unsqueeze(torch.cat([energy, position]), 0)
    #         }
    #     }
    #
    #     return featurized_obs

    def get_state(self, last_3_actions):
        obs = self.state

        player_channel = np.zeros((4, obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1), dtype=float)
        obstacle_1 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        obstacle_random = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        obstacle_5 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        obstacle_10 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        obstacle_20 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        obstacle_40 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        obstacle_100 = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        obstacle_value_min = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        obstacle_value_max = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)

        gold = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)
        gold_amount = np.zeros([obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1], dtype=float)

        for i in range(obs.mapInfo.max_y + 1):
            for j in range(obs.mapInfo.max_x + 1):
                type, value = None, None
                for cell in obs.mapInfo.obstacles:
                    if j == cell["posx"] and i == cell["posy"]:
                        type, value = cell["type"], cell["value"]

                if type is None and value is None:
                    has_gold = False
                    for cell in obs.mapInfo.golds:
                        if j == cell["posx"] and i == cell["posy"]:
                            has_gold = True

                    if not has_gold:
                        value = -1

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

        player_channel[0][obs.y, obs.x] = 1

        id = 1
        for player in obs.players:
            if player["playerId"] == obs.id:
                continue

            if "status" in player and player["status"] == constants.Status.STATUS_PLAYING.value:
                player_channel[id][player["posy"], player["posx"]] = 1
                id += 1

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

        position = np.clip(np.array([obs.y / 8 * 2 - 1,
                                     obs.x / 20 * 2 - 1]), -1, 1)

        one_hot_last_3_actions = np.zeros((3, 6), dtype=np.float32)
        one_hot_last_3_actions[np.arange(3), last_3_actions] = 1
        one_hot_last_3_actions = one_hot_last_3_actions.reshape(-1)

        featurized_obs = {
            "obs": {
                "conv_features": torch.unsqueeze(torch.tensor(np.concatenate([
                    player_channel,
                    board,
                    np.full((1, obs.mapInfo.max_y + 1, obs.mapInfo.max_x + 1),
                            fill_value=max(0, obs.energy / (constants.MAX_ENERGY / 2)))
                ]), dtype=torch.float), 0),
                "fc_features": torch.unsqueeze(torch.tensor(np.concatenate([
                        position,
                        one_hot_last_3_actions
                    ]), dtype=torch.float), 0)
            }
        }

        return featurized_obs, self.state

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING
