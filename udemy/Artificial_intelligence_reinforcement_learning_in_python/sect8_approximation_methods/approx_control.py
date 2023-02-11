#!/bin/env python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/learn/lecture/26536132


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, GridState
from tqdm import tqdm
from typing import Union
from utils import epsilon_greedy_by_model, print_values, print_policy, ALL_POSSIBLE_ACTIONS
from sklearn.kernel_approximation import Nystroem, RBFSampler

GAMMA = 0.9
ALPHA = 0.1

ACTION_2_INT = {a: i for i, a in enumerate(ALL_POSSIBLE_ACTIONS)}
INT_2_ONEHOT = np.eye(len(ALL_POSSIBLE_ACTIONS))


def one_hot(k):
  return INT_2_ONEHOT[k]


def merge_state_action(s: Union[tuple[int, int], GridState], a):
  if isinstance(s, GridState):
    s = (s.i, s.j)

  ai = one_hot(ACTION_2_INT[a])
  return np.concatenate((s, ai))


def gather_samples(grid, n_episodes=1000):
  samples = []
  for _ in range(n_episodes):
    s = grid.reset()
    while not grid.game_over():
      a = np.random.choice(ALL_POSSIBLE_ACTIONS)
      sa = merge_state_action(s, a)
      samples.append(sa)

      r = grid.move(a)
      s = grid.current_state

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

  def predict(self, s: Union[tuple[int,int], GridState], a):
    if isinstance(s, GridState):
      s = (s.i, s.j)

    sa = merge_state_action(s, a)
    x = self.featurizer.transform([sa])[0]
    return x @ self.w

  def predict_all_actions(self, s: Union[tuple[int,int], GridState]):
    if isinstance(s, GridState):
      s = (s.i, s.j)

    return [self.predict(s, a) for a in ALL_POSSIBLE_ACTIONS]

  def grad(self, s: Union[tuple[int,int], GridState], a):
    if isinstance(s, GridState):
      s = (s.i, s.j)

    sa = merge_state_action(s, a)
    x = self.featurizer.transform([sa])[0]
    return x


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  # grid = standard_grid()
  grid = negative_grid(step_cost=-0.1)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)
  print("")

  model = Model(grid)
  reward_per_episode = []
  state_visit_count = {}

  # repeat until convergence
  n_episodes = 20000
  for it in tqdm(range(n_episodes)):
    # if (it + 1) % 100 == 0:
    #  print(it + 1)

    s = grid.reset()
    state_visit_count[s] = state_visit_count.get(s, 0) + 1
    episode_reward = 0
    while not grid.game_over():
      a = epsilon_greedy_by_model(model, s)
      r = grid.move(a)
      s2 = grid.current_state
      state_visit_count[s2] = state_visit_count.get(s2, 0) + 1

      # get the target
      if grid.game_over():
        target = r
      else:
        values = model.predict_all_actions(s2)
        target = r + GAMMA * np.max(values)

      # update the model
      g = model.grad(s, a)
      err = target - model.predict(s, a)
      model.w += ALPHA * err * g

      # accumulate reward
      episode_reward += r

      # update state
      s = s2

    reward_per_episode.append(episode_reward)

  plt.plot(reward_per_episode)
  plt.title("Reward per episode")
  plt.show()

  # obtain V* and pi*
  V = {}
  greedy_policy = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      values = model.predict_all_actions(s)
      V[s] = np.max(values)
      greedy_policy[s] = ALL_POSSIBLE_ACTIONS[np.argmax(values)]
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0

  print("\nvalues:")
  print_values(V, grid)
  print("\npolicy:")
  print_policy(greedy_policy, grid)

  print("\nstate_visit_count:")
  state_sample_count_arr = np.zeros((grid.rows, grid.cols))
  for i in range(grid.rows):
    for j in range(grid.cols):
        state_sample_count_arr[i,j] = state_visit_count.get(GridState(i, j), 0)

  df = pd.DataFrame(state_sample_count_arr)
  print(df)
