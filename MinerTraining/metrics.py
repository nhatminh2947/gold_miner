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
    FINDING_GOLD = 9
    DOUBLE_FREE = 10
    TRIPLE_FREE = 11
