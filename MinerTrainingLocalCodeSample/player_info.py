import constants


class PlayerInfo:
    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = constants.Status.STATUS_PLAYING.value
        self.freeCount = 0

    def reset(self, x, y, energy):
        self.score = 0
        self.energy = energy
        self.posx = x
        self.posy = y
        self.lastAction = -1
        self.status = constants.Status.STATUS_PLAYING.value
        self.freeCount = 0
