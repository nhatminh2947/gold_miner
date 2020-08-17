import json

from . import GameInfo


class UserMatch:
    def __init__(self):
        self.playerId = 1
        self.posx = 0
        self.posy = 0
        self.energy = 50
        self.gameinfo = GameInfo()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
