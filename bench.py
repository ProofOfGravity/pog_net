from enum import Enum

class Action(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    NO_OP = 5


print(Action(0))


