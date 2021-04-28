"""
The World map
"""

from enum import IntEnum

import numpy as np

from contracts import contract, new_contract

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

cell_only = new_contract('cell_only', lambda c: isinstance(c, Cell))
numpy_bool = new_contract('numpy_bool', lambda c: isinstance(c, np.bool_))

class Cell:
    def __init__(self, x, y):
        if type(x) != int or x < 0:
            raise ValueError('`x` should a positive integer: got {!r}'.format(x))
        if type(y) != int or y < 0:
            raise ValueError('`y` should a positive integer: got {!r}'.format(y))

        self._x = x
        self._y = y

    @contract
    def __add__(self, other):
        """
            :param other: cell to add
            :type other: cell_only

            :return: the addition of both cells
            :rtype: cell_only
        """
        x = self.x + other.x
        y = self.y + other.y
        return Cell(x, y)

    @contract
    def __eq__(self, other):
        """
            :param other: cell to add
            :type other: cell_only

            :return: whether both are equal
            :rtype: bool | numpy_bool
        """
        return self.x == other.x and self.y == other.y

    @contract
    def __hash__(self) -> 'int':
        return hash((self.x,self.y))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

cell = new_contract('cell', lambda c: isinstance(c, Cell) and c.x >= 0 and c.y >= 0)

class World:

    def __init__(self, worldMap):
        if type(worldMap) != np.ndarray or len(worldMap.shape) != 2:
            raise ValueError('`map` should be numpy array of one dimension: got {!r}'.format(worldMap))

        self._worldMap = worldMap
        self._height = len(worldMap)
        self._width = len(worldMap[0])

    def isEmpty(self):
        return len(self._worldMap) == 0

    @contract
    def isObstacle(self, pos):
        """
            :param pos: position
            :type pos: cell

            :return: whether an obstacle is at pos
            :rtype: bool | numpy_bool
        """
        return self._worldMap[pos.x][pos.y] == CellType.OBSTACLE

    @contract
    def isFree(self, pos):
        """
            :param pos: position
            :type pos: cell

            :return: whether nothing is at pos
            :rtype: bool | numpy_bool
        """
        return self._worldMap[pos.x][pos.y] == CellType.FREE

    @contract
    def getDisplay(self, pos):
        """
            :param pos: position
            :type pos: cell

            :return: the character for the cell
            :rtype: str
        """
        if self._worldMap[pos.x][pos.y] == CellType.FREE:
            return ' '
        if self._worldMap[pos.x][pos.y] == CellType.OBSTACLE:
            return '#'
        raise ValueError('Unknown cell type in world at {!r}'.format(pos))

    @contract
    def getValue(self, pos):
        """
            :param pos: position
            :type pos: cell

            :return: the value at cell pos
            :rtype: int,>=0
        """
        return self._worldMap[pos.x][pos.y]

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width