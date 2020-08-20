from enum import Enum


class Metrics(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    FREE = 4
    CRAFT = 5
    ENERGY = 6
    INVALID_CRAFT = 7
    INVALID_FREE = 8
