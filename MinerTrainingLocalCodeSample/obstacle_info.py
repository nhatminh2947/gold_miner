class ObstacleInfo:
    # initial energy for obstacles: Land (key = 0): -1, Forest(key = -1): 0 (random), Trap(key = -2): -10, Swamp (key = -3): -5
    types = {0: -1, -1: 0, -2: -10, -3: -5}

    def __init__(self):
        self.type = 0
        self.posx = 0
        self.posy = 0
        self.value = 0
