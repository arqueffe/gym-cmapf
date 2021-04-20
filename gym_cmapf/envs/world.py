"""
The World map
"""

from enum import IntEnum

import numpy as np

"""
Setting of the map
X\Y   0 1 2
    0 # # #
    1 # # #
    2 # # #
"""

class CellType(IntEnum):
    FREE        = 0
    OBSTACLE    = 1

class World:

    def __init__(self, worldMap):
        if type(worldMap) != np.ndarray or len(worldMap.shape) != 2:
            raise ValueError('`map` should be numpy array of one dimension: got {!r}'.format(worldMap))

        self._worldMap = worldMap
        self._height = len(worldMap)
        self._width = len(worldMap[0])

    def isEmpty(self):
        return len(self._worldMap) == 0;

    def isObstacle(self, pos):
        return self._worldMap[pos[0]][pos[1]] == CellType.OBSTACLE

    def isFree(self, pos):
        return self._worldMap[pos[0]][pos[1]] == CellType.FREE

    def getDisplay(self, pos):
        if self._worldMap[pos[0]][pos[1]] == CellType.FREE:
            return ' '
        if self._worldMap[pos[0]][pos[1]] == CellType.OBSTACLE:
            return '#'
        print(self._worldMap[pos[0]][pos[1]])

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width