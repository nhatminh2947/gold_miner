from enum import Enum

from gym.spaces import Dict, Box, Discrete

E_LOSS = [-1, 0, -10, -5, -20, -40, -100, -4]

# GOLD PER MAP
# Map 1 6500 - 20 gold spots
# Map 2 6850 - 19 gold spots
# Map 3 7300 - 23 gold spots
# Map 4 7500 - 19 gold spots
# Map 5 7100 - 21 gold spots
MAX_LEN = 100
NUM_FEATURES = 16
MAX_EXTRACTABLE_GOLD = 5000
N_COLS = 21
N_ROWS = 9
MAX_ENERGY = 100
BASE_ENERGY = 5

OBS_SPACE = Dict({
    "conv_features": Box(low=-1, high=2, shape=(NUM_FEATURES, N_ROWS, N_COLS)),
    "fc_features": Box(low=-1, high=2, shape=(28,))
})

ACT_SPACE = Discrete(6)

GOLD_PROB = [0.1, 0.1, 0.06, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04,
             0.04728, 0.04728, 0.04728, 0.04728, 0.04728, 0.0394, 0.0394, 0.0394, 0.0394,
             0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

GOLD_VALUE = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
              550, 600, 650, 700, 750, 800, 850, 900, 950,
              1000, 1050, 1100, 1150, 1200, 1250]


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
