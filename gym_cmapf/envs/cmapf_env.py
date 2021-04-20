"""
The CMAPF OpenAI gym environment
"""

from enum import IntEnum

import gym
from gym import spaces

import numpy as np

from contracts import contract, ContractsMeta, with_metaclass
from abc import abstractmethod
from world import World

ACTION_COST   = -0.3
IDLE_COST     = -0.5
GOAL_REWARD   = 0.0
FINISH_REWARD = 20.0

class Action(IntEnum):
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

# Direction : (X,Y)
actionToDirection = {
  Action.IDLE:  (0,0),
  Action.UP:    (-1,0),
  Action.DOWN:  (1,0),
  Action.LEFT:  (0,-1),
  Action.RIGHT: (0,1)
}

directionToAction = {
  dir:act for act,dir in actionToDirection.items()
}

class Agent(with_metaclass(ContractsMeta, object)):

  def __init__(self, id):
    if id != int(id) or id < 0:
      raise ValueError('`id` should be a positive integer: got {!r}'.format(id))

    self._id = id

  @abstractmethod
  @contract
  def observe(self, state : 'array'):
    pass

  @contract
  def reward(self, prevState : 'array', nextState : 'array') -> float:
    pass

  @property
  def id(self):
    return self._id

class BasicAgent(Agent):

  def __init__(self, id):
      super(BasicAgent, self).__init__(id)

  def observe(self, state):
      return state[self.id]

  def reward(self, prevState, nextState):
      return 0.0

class CMAPFEnv(gym.Env):

  metadata = {
    'render.modes': ['console']
  }

  def __init__(self, num_agents, world, starts, goals, agents):

    if num_agents != int(num_agents) or num_agents <= 0:
      raise ValueError('`num_agents` should be a positive integer: got {!r}'.format(num_agents))

    if type(world) != World or world.isEmpty():
      raise ValueError('`world` should be a non-empty World: got {!r}'.format(world))

    if type(starts) != np.ndarray or len(starts) != num_agents or len(starts.shape) != 2:
      raise ValueError('`starts` should be numpy array of one dimension: got {!r}'.format(starts))

    if type(goals) != np.ndarray or len(goals) != num_agents or len(goals.shape) != 2:
      raise ValueError('`goals` should be numpy array of one dimension: got {!r}'.format(goals))

    if type(agents) != list or len(agents) != num_agents or not all(isinstance(x, Agent) for x in agents):
      raise ValueError('`agents` should be list of Agent: got {!r}'.format(agents))

    self._num_agents = num_agents
    self._world = world
    self._starts = starts
    self._goals = goals
    self._agents = agents

    self.current = starts

  def step(self, action):

    state = self.current.copy()

    if type(action) != list or len(action) != self._num_agents or not all(isinstance(x, Agent) for x in self._agents):
      raise ValueError('`action` should be list of actions: got {!r}'.format(action))

    for agt in self._agents:
      pos = self.current[agt.id] + actionToDirection[action[agt.id]]
      if pos[0] >= 0 and pos[0] < self._world.height and pos[1] >= 0 and pos[1] < self._world.width:
        if self._world.isFree(pos):
          state[agt.id] = pos

    obs = [agt.observe(state) for agt in self._agents]

    rewards = [agt.reward(self.current, state) for agt in self._agents]

    done = [self._goals[agt.id][0] == state[agt.id][0] and self._goals[agt.id][1] == state[agt.id][1] for agt in self._agents]

    self.current = state

    return self.current, obs, rewards, done

  def reset(self):
    self.current = self._starts

  def render(self, mode='console'):

    if mode == 'console':

      display = ""
      done = False
      for x in range(0, self._world.height):
        for y in range(0, self._world.width):
          for agt in self._agents:
            if x == self.current[agt.id][0] and y == self.current[agt.id][1]:
              display += str(agt.id)
              done = True
              break
          if not done:
            display += self._world.getDisplay((x,y))
          done = False
        if x != self._world.height -1:
          display+='\n'

      print(display)

  def close(self):
    super(gym.Env, self).close()



if __name__ == "__main__":
  global charCount
  charCount = 0
  m = np.array([[1,1,1],[0,0,0],[1,0,1]])
  w = World(m)
  s = np.array([(1,1)])
  nb = 1
  cmapf = CMAPFEnv(nb, w, s, np.array([(1,1)]), [BasicAgent(0)])
  while(1):
    cmapf.render()
    l = []
    for agt in range(0, nb):
      valid = False
      intInp = -1
      while not valid:
        inp = input('Action of agent {}: '.format(agt))
        try:
          intInp=int(inp)
          if intInp < 0 or intInp > 4:
            print("An action is an integer between 0 and 4")
          else:
            valid = True
        except ValueError:
          print("An action is an integer between 0 and 4")
      l.append(intInp)
    cmapf.step(l)