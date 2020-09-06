import numpy as np

import constants
from .game_socket_dummy import GameSocket  # in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from .miner_state import State

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
            mapID = np.random.randint(0, 13)
            # version = np.random.randint(0, 4)
            # mapID = 0
            # version = 0
            # utils.generate_map()
            posID_x = np.random.randint(constants.N_COLS)
            posID_y = np.random.randint(constants.N_ROWS)
            request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
            # request = f"map{mapID}_{version},{posID_x},{posID_y},50,100"
            # Send the request to the game environment (GAME_SOCKET_DUMMY.py)
            self.send_map_info(request)

            message = self.socket.receive()  # receive game info from server
            self.state.init_state(message)  # init state
            return self.state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, actions):  # step process
        self.socket.send(actions)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            self.state.update_state(message)  # update to local state
            return self.state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.width + 1, self.state.mapInfo.height + 1], dtype=int)
        for i in range(self.state.mapInfo.width + 1):
            for j in range(self.state.mapInfo.height + 1):
                if self.state.mapInfo.get_obstacle_type(i, j) == TreeID:  # Tree
                    view[i, j] = -TreeID
                if self.state.mapInfo.get_obstacle_type(i, j) == TrapID:  # Trap
                    view[i, j] = -TrapID
                if self.state.mapInfo.get_obstacle_type(i, j) == SwampID:  # Swamp
                    view[i, j] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = self.state.mapInfo.gold_amount(i, j)

        DQNState = view.flatten().tolist()  # Flattening the map matrix to a vector

        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)
        # Add position of bots
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                DQNState.append(player["posx"])
                DQNState.append(player["posy"])

        # Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)

        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        if score_action > 0:
            # If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action

        # If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle_type(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= TreeID
        if self.state.mapInfo.get_obstacle_type(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= TrapID
        if self.state.mapInfo.get_obstacle_type(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= SwampID

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -10

        # Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -10
        # print ("reward",reward)
        return reward

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
