#!/bin/env python3
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/learn/lecture/6431040


import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, GridState
from utils import epsilon_greedy, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  policy = {
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

  # initialize V(s) and returns
  V = {}
  states = grid.all_states()
  for s in states:
    V[s] = 0

  # store max change in V(s) per episode
  deltas = []

  # repeat until convergence
  n_episodes = 10000
  for it in range(n_episodes):
    # begin a new episode
    s = grid.reset()

    delta = 0
    while not grid.game_over():
      a = epsilon_greedy(policy, s)

      r = grid.move(a)
      s_next = grid.current_state

      # update V(s)
      v_old = V[s]
      V[s] = V[s] + ALPHA * (r + GAMMA * V[s_next] - V[s])
      delta = max(delta, np.abs(V[s] - v_old))

      # next state becomes current state
      s = s_next

    # store delta
    deltas.append(delta)

  plt.plot(deltas)
  plt.show()

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
