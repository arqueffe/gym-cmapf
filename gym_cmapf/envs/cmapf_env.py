"""
The CMAPF OpenAI gym environment
"""

from enum import IntEnum

import gym
from gym import spaces

import numpy as np
import pygame

from contracts import contract, ContractsMeta, with_metaclass
from abc import abstractmethod
from world import World, Cell

# Penalities
ACTION_COST         = -0.3
IDLE_COST           = -0.5
COLLISION_COST      = -2.0
DISCONNECTION_COST  = -2.0
# Rewards
GOAL_REWARD     = 0.0
FINISH_REWARD   = 20.0

COLLISION_FORBIDDEN = True
DISCONNECTION_FORBIDDEN = True

SCREEN_SIZE = (600,600)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class ActionID(IntEnum):
  IDLE = 0
  UP = 1
  DOWN = 2
  LEFT = 3
  RIGHT = 4

"""
Setting of the map
X\Y   0 1 2
    0 # # #
    1 # # #
    2 # # #
"""

class Action(Cell):
  def __init__(self, x, y):
        self._x = x
        self._y = y

# Direction : (X,Y)
actionToDirection = {
  ActionID.IDLE:  Action(0,0),
  ActionID.UP:    Action(-1,0),
  ActionID.DOWN:  Action(1,0),
  ActionID.LEFT:  Action(0,-1),
  ActionID.RIGHT: Action(0,1)
}

directionToAction = {
  dir:act for act,dir in actionToDirection.items()
}




class Agent(with_metaclass(ContractsMeta, object)):

  def __init__(self, id):
    if id != int(id) or id < 0:
      raise ValueError('`id` should be a positive integer: got {!r}'.format(id))

    self._id = id
    self._color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

  @abstractmethod
  @contract
  def observe(self, state : 'array'):
    pass

  @abstractmethod
  @contract
  def reward(self, prevState : 'array', nextState : 'array') -> float:
    pass

  @property
  def id(self):
    return self._id

  @property
  def color(self):
    return self._color

class BasicAgent(Agent):

  def __init__(self, id):
      super(BasicAgent, self).__init__(id)

  def observe(self, state):
      return state[self.id]

  def reward(self, prevState, nextState):
      return 0.0

class CMAPFEnv(gym.Env):

  metadata = {
    'render.modes': ['console', 'pygame']
  }

  def __init__(self, num_agents, world, starts, goals, agents):

    # Type and Value checking
    if num_agents != int(num_agents) or num_agents <= 0:
      raise ValueError('`num_agents` should be a positive integer: got {!r}'.format(num_agents))

    if type(world) != World or world.isEmpty():
      raise ValueError('`world` should be a non-empty World: got {!r}'.format(world))

    if type(starts) != np.ndarray or len(starts) != num_agents or len(starts.shape) != 1:
      raise ValueError('`starts` should be numpy array of one dimension: got {!r}'.format(starts))

    if type(goals) != np.ndarray or len(goals) != num_agents or len(goals.shape) != 1:
      raise ValueError('`goals` should be numpy array of one dimension: got {!r}'.format(goals))

    if type(agents) != list or len(agents) != num_agents or not all(isinstance(x, Agent) for x in agents):
      raise ValueError('`agents` should be list of Agent: got {!r}'.format(agents))

    # Initialization
    self._num_agents  = num_agents
    self._world       = world
    self._starts      = starts
    self._goals       = goals
    self._agents      = agents

    self._current     = starts
    self._pygameReady = False

  @contract
  def step(self, jointAction):
    """
        :param jointAction: joint action of the agents
        :type jointAction: list[>0](int)

        :return: global state, observations, rewards, done status
        :rtype: tuple(array,list[N],list[N],list[N]),N>0
    """
    state = self._current.copy()

    if type(jointAction) != list or len(jointAction) != self._num_agents or not all(isinstance(x, Agent) for x in self._agents):
      raise ValueError('`action` should be list of actions: got {!r}'.format(jointAction))

    for agt in self._agents:
      pos = self._current[agt.id] + actionToDirection[jointAction[agt.id]]
      if pos.x >= 0 and pos.x < self._world.height and pos.y >= 0 and pos.y < self._world.width:
        if self._world.isFree(pos):
          state[agt.id] = pos

    col = []

    if COLLISION_FORBIDDEN:
      for a in self._agents:
        for b in self._agents:
          if a.id >= b.id:
            continue
          # Position Collision
          if state[a.id] == state[b.id]:
            state[a.id] = self._current[a.id]
            state[b.id] = self._current[b.id]
            col.append([a,b])
          # Head-on Collision
          elif state[a.id] == self._current[b.id] and state[b.id] == self._current[a.id]:
            state[a.id] = self._current[a.id]
            state[b.id] = self._current[b.id]
            col.append([a,b])

    disc = False

    if DISCONNECTION_FORBIDDEN:
      if not self._world.isConnected(state):
        disc = True
        state = self._current

    obs = [agt.observe(state) for agt in self._agents]

    rewards = [agt.reward(self._current, state) for agt in self._agents]

    done = [self._goals[agt.id].x == state[agt.id].x and self._goals[agt.id].y == state[agt.id].y for agt in self._agents]

    self._current = state

    return self._current, obs, rewards, done

  def reset(self):
    self._current = self._starts

  def render(self, mode='console'):
    if mode == 'console':
      self.__renderToConsole()
    if mode == 'pygame':
      self.__renderToPygame()

  def __renderToConsole(self):
      display = ""
      done = False
      for x in range(0, self._world.height):
        for y in range(0, self._world.width):
          for agt in self._agents:
            if x == self._current[agt.id].x and y == self._current[agt.id].y:
              display += str(agt.id)
              done = True
              break
          if not done:
            display += self._world.getDisplay(Cell(x,y))
          done = False
        if x != self._world.height -1:
          display+='\n'
      print(display)

  def __renderToPygame(self):
    if not self._pygameReady:
      pygame.init()
      pygame.display.set_caption('CMAPF Renderer')
      self._screen = pygame.display.set_mode(SCREEN_SIZE)
      self._background = pygame.Surface(self._screen.get_size()).convert()
      self._background.fill((255, 255, 255))
      self._pygameReady = True
    ## Draw the maze
    widthScale = self._screen.get_size()[1] / self._world.width
    heightScale = self._screen.get_size()[0] / self._world.height
    scale = max(widthScale, heightScale)
    font = pygame.font.SysFont(None, int(scale/2))
    for x in range(0, self._world.height):
        for y in range(0, self._world.width):
          pygame.draw.rect(self._screen, self._world.getColor(Cell(x,y)), (y*scale, x*scale, (y+1)*scale, (x+1)*scale))
          for agt in self._agents:
            if x == self._current[agt.id].x and y == self._current[agt.id].y:
              pygame.draw.circle(self._screen, agt.color, (y*scale + scale/2, x*scale + scale/2), scale/4)
              img = font.render(str(agt.id), True, BLACK)
              self._screen.blit(img, (y*scale + scale/3, x*scale + scale/3))
    pygame.display.flip()

  def close(self):
    super(gym.Env, self).close()

if __name__ == "__main__":
  m = np.array([[0,1,1,0],[0,0,0,0],[1,0,1,1],[1,0,0,0]])
  w = World(m, 2)
  s = np.array([Cell(1,1), Cell(1,0)])
  t = np.array([Cell(1,1), Cell(2,0)])
  nb = 2
  cmapf = CMAPFEnv(nb, w, s, t, [BasicAgent(0),BasicAgent(1)])
  close = False
  while not close:
    cmapf.render('pygame')
    l = []
    for agt in range(0, nb):
      valid = False
      intInp = -1
      while not valid and not close:
        inp = input('Action of agent {}: '.format(agt))
        try:
          intInp=int(inp)
          print(str(intInp))
          if intInp < -1 or intInp > 4:
            print("An action is an integer between 0 and 4. (-1 to quit)")
          else:
            if intInp == -1:
              close = True
            else:
              valid = True
        except ValueError:
          print("An action is an integer between 0 and 4. (-1 to quit)")
      if valid:
        l.append(intInp)
    if not close:
      cmapf.step(l)