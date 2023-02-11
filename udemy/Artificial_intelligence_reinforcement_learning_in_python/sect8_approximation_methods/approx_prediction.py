#!/bin/env python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# Course link: https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/learn/lecture/26536102

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, GridState
from sklearn.kernel_approximation import Nystroem, RBFSampler
from tqdm import tqdm
from typing import Union
from utils import epsilon_greedy, print_values, print_policy, ALL_POSSIBLE_ACTIONS


GAMMA = 0.9
ALPHA = 0.01


def gather_samples(grid, n_episodes=10000):
  samples = []
  for _ in range(n_episodes):
    s = grid.reset()
    samples.append(s)
    while not grid.game_over():
      a = np.random.choice(ALL_POSSIBLE_ACTIONS)
      r = grid.move(a)
      s = grid.current_state
      samples.append(s)

  samples = list(map(lambda s: (s.i, s.j), samples))
  return samples


class Model:
  def __init__(self, grid):
    # fit the featurizer to data
    samples = gather_samples(grid)
    # self.featurizer = Nystroem()
    self.featurizer = RBFSampler()
    self.featurizer.fit(samples)
    dims = self.featurizer.n_components

    # initialize linear model weights
    self.w = np.zeros(dims)

  def predict(self, s: Union[tuple[int, int], GridState]):
    if isinstance(s, GridState):
      s = (s.i, s.j)

    x = self.featurizer.transform([s])[0]
    return x @ self.w

  def grad(self, s: Union[tuple[int, int], GridState]):
    if isinstance(s, GridState):
      s = (s.i, s.j)

    x = self.featurizer.transform([s])[0]
    return x


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  greedy_policy = {
    GridState(2, 0): 'U',
    GridState(1, 0): 'U',
    GridState(0, 0): 'R',
    GridState(0, 1): 'R',
    GridState(0, 2): 'R',
    GridState(1, 2): 'R',
    GridState(2, 1): 'R',
    GridState(2, 2): 'R',
    GridState(2, 3): 'U',
  }

  model = Model(grid)
  mse_per_episode = []

  # repeat until convergence
  n_episodes = 10000
  for it in tqdm(range(n_episodes)):
    # if (it + 1) % 100 == 0:
    #   print(it + 1)

    s = grid.reset()
    Vs = model.predict(s)
    n_steps = 0
    episode_err = 0
    while not grid.game_over():
      a = epsilon_greedy(greedy_policy, s)
      r = grid.move(a)
      s2 = grid.current_state

      # get the target
      if grid.is_terminal(s2):
        target = r
      else:
        Vs2 = model.predict(s2)
        target = r + GAMMA * Vs2

      # update the model
      g = model.grad(s)
      err = target - Vs
      model.w += ALPHA * err * g

      # accumulate error
      n_steps += 1
      episode_err += err*err

      # update state
      s = s2
      Vs = Vs2

    mse = episode_err / n_steps
    mse_per_episode.append(mse)

  plt.plot(mse_per_episode)
  plt.title("MSE per episode")
  plt.show()

  # obtain predicted values
  V = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      V[s] = model.predict(s)
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0

  print("\nvalues:")
  print_values(V, grid)
  print("\npolicy:")
  print_policy(greedy_policy, grid)
