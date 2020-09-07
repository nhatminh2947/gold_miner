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

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
