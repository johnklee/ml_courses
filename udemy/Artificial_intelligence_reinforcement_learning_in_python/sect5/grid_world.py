# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division, annotations
from builtins import range

import dataclasses
import enum
import numpy as np
import random


@dataclasses.dataclass
class GridState:
  i: int
  j: int

  def copy(self) -> GridState:
    return GridState(i=self.i, j=self.j)

  def __hash__(self):
    return hash((self.i, self.j))


class GridAction(enum.Enum):
  UP='U'
  DOWN='D'
  LEFT='L'
  RIGHT='R'


ACTION_SPACE = ('U', 'D', 'L', 'R')


class Grid: # Environment
  def __init__(self, rows: int, cols: int, init_state: GridState):
    self.rows = rows
    self.cols = cols
    self._begin_state = init_state
    self.state = self._begin_state.copy()

  def set(self, rewards, actions):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s: GridState):
    self.state = s

  @property
  def current_state(self) -> GridState:
    return self.state

  def is_terminal(self, s: GridState):
    return s not in self.actions

  def reset(self) -> GridState:
    # put agent back in start position
    self.state = self._begin_state.copy()
    return self.state

  def get_next_state(self, s: GridState, a) -> GridState:
    # if this action moves you somewhere else, then it will be in this dictionary
    next_state = s.copy()
    if a in self.actions[s]:
      if a == 'U':
        next_state.i -= 1
      elif a == 'D':
        next_state.i += 1
      elif a == 'R':
        next_state.j += 1
      elif a == 'L':
        next_state.j -= 1

    return next_state

  def random_move(self) -> float:
    # Take random move at current state
    next_action = random.choice(self.actions[self.current_state])
    return self.move(next_action)

  def move(self, action) -> float:
    # check if legal move first
    if action in self.actions[self.state]:
      if action == 'U':
        self.state.i -= 1
      elif action == 'D':
        self.state.i += 1
      elif action == 'R':
        self.state.j += 1
      elif action == 'L':
        self.state.j -= 1

    # return a reward (if any)
    return self.rewards.get(self.state, 0)

  def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
    if action == 'U':
      self.i += 1
    elif action == 'D':
      self.i -= 1
    elif action == 'R':
      self.j -= 1
    elif action == 'L':
      self.j += 1
    # raise an exception if we arrive somewhere we shouldn't be
    # should never happen
    assert(self.current_state in self.all_states())

  def game_over(self) -> bool:
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return self.state not in self.actions

  def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
    return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
  g = Grid(3, 4, GridState(i=2, j=0))
  rewards = {GridState(i=0, j=3): 1, GridState(i=1, j=3): -1}
  actions = {
    GridState(0, 0): ('D', 'R'),
    GridState(0, 1): ('L', 'R'),
    GridState(0, 2): ('L', 'D', 'R'),
    GridState(1, 0): ('U', 'D'),
    GridState(1, 2): ('U', 'D', 'R'),
    GridState(2, 0): ('U', 'R'),
    GridState(2, 1): ('L', 'R'),
    GridState(2, 2): ('L', 'R', 'U'),
    GridState(2, 3): ('L', 'U'),
  }
  g.set(rewards, actions)
  return g


class WindyGrid:
  def __init__(self, rows, cols, start):
    self.rows = rows
    self.cols = cols
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions, probs):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions
    self.probs = probs

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    s = (self.i, self.j)
    a = action

    next_state_probs = self.probs[(s, a)]
    next_states = list(next_state_probs.keys())
    next_probs = list(next_state_probs.values())
    next_state_idx = np.random.choice(len(next_states), p=next_probs)
    s2 = next_states[next_state_idx]

    # update the current state
    self.i, self.j = s2

    # return a reward (if any)
    return self.rewards.get(s2, 0)

  def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return (self.i, self.j) not in self.actions

  def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
    return set(self.actions.keys()) | set(self.rewards.keys())


def windy_grid():
  g = WindyGrid(3, 4, (2, 0))
  rewards = {(0, 3): 1, (1, 3): -1}
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }

  # p(s' | s, a) represented as:
  # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
  probs = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
  g.set(rewards, actions, probs)
  return g


def windy_grid_no_wind():
  g = windy_grid()
  g.probs[((1, 2), 'U')] = {(0, 2): 1.0}
  return g


def windy_grid_penalized(step_cost=-0.1):
  g = WindyGrid(3, 4, (2, 0))
  rewards = {
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (0, 3): 1,
    (1, 3): -1
  }
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }

  # p(s' | s, a) represented as:
  # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
  probs = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
  g.set(rewards, actions, probs)
  return g


def grid_5x5(step_cost=-0.1):
  g = Grid(5, 5, (4, 0))
  rewards = {GridState(0, 4): 1, GridState(1, 4): -1}
  actions = {
    GridState(0, 0): ('D', 'R'),
    GridState(0, 1): ('L', 'R'),
    GridState(0, 2): ('L', 'R'),
    GridState(0, 3): ('L', 'D', 'R'),
    GridState(1, 0): ('U', 'D', 'R'),
    GridState(1, 1): ('U', 'D', 'L'),
    GridState(1, 3): ('U', 'D', 'R'),
    GridState(2, 0): ('U', 'D', 'R'),
    GridState(2, 1): ('U', 'L', 'R'),
    GridState(2, 2): ('L', 'R', 'D'),
    GridState(2, 3): ('L', 'R', 'U'),
    GridState(2, 4): ('L', 'U', 'D'),
    GridState(3, 0): ('U', 'D'),
    GridState(3, 2): ('U', 'D'),
    GridState(3, 4): ('U', 'D'),
    GridState(4, 0): ('U', 'R'),
    GridState(4, 1): ('L', 'R'),
    GridState(4, 2): ('L', 'R', 'U'),
    GridState(4, 3): ('L', 'R'),
    GridState(4, 4): ('L', 'U'),
  }
  g.set(rewards, actions)

  # non-terminal states
  visitable_states = actions.keys()
  for s in visitable_states:
    g.rewards[s] = step_cost

  ieturn 
