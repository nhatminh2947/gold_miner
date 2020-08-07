from Miner_Training_Local_CodeSample import PlayerInfo
from Miner_Training_Local_CodeSample.MINER_STATE import State


class BaseAgent:
    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)

    def next_action(self):
        raise NotImplemented

    def new_game(self, data):
        try:
            self.state.init_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def new_state(self, data):
        # action = self.next_action();
        # self.socket.send(action)
        try:
            self.state.update_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
