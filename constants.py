from enum import Enum

from gym.spaces import Dict, Box, Discrete

# GOLD PER MAP
# Map 1 6500 - 20 gold spots
# Map 2 6850 - 19 gold spots
# Map 3 7300 - 23 gold spots
# Map 4 7500 - 19 gold spots
# Map 5 7100 - 21 gold spots
MAX_LEN = 100
NUM_FEATURES = 13
MAX_EXTRACTABLE_GOLD = 5000
N_COLS = 21
N_ROWS = 9
MAX_ENERGY = 100
BASE_ENERGY = 5

OBS_SPACE = Dict({
    "conv_features": Box(low=0, high=2, shape=(NUM_FEATURES, N_ROWS, N_COLS)),
    "fc_features": Box(low=-1, high=1, shape=(4 + 2 + 4,))
})

ACT_SPACE = Discrete(6)


class Obstacle(Enum):
    LAND = 0
    TREE = 1
    TRAP = 2
    SWAMP = 3


class Action(Enum):
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5


class Status(Enum):
    STATUS_PLAYING = 0
    STATUS_ELIMINATED_WENT_OUT_MAP = 1
    STATUS_ELIMINATED_OUT_OF_ENERGY = 2
    STATUS_ELIMINATED_INVALID_ACTION = 3
    STATUS_STOP_EMPTY_GOLD = 4
    STATUS_STOP_END_STEP = 5
