from enum import Enum

from gym.spaces import Dict, Box, Discrete

# GOLD PER MAP
# Map 1 6500
# Map 2 6850
# Map 3 7300
# Map 4 7500
# Map 5 7100
MAX_LEN = 100
NUM_FEATURES = 15
N_COLS = 21
N_ROWS = 9
MAX_GOLD = 10000
MAX_ENERGY = 100
BASE_REWARD = 50 / MAX_GOLD  # 0.005
BASE_ENERGY = 5
SCALE = 1 / (BASE_REWARD * 100)

OBS_SPACE = Dict({
    "conv_features": Box(low=0, high=2, shape=(NUM_FEATURES, N_ROWS, N_COLS)),
    "fc_features": Box(low=-1, high=1, shape=(3,))
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
