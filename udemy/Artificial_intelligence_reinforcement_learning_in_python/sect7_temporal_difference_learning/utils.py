"""Common utility used in RL course."""
import numpy as np
from grid_world import GridState


ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often

  # find max val
  max_val = max(d.values())

  # find keys corresponding to max val
  max_keys = [key for key, val in d.items() if val == max_val]

  return np.random.choice(max_keys), max_val


def epsilon_greedy_by_qtable(Q, s, eps=0.1):
  if np.random.random() < eps:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)
  else:
    a_opt = max_dict(Q[s])[0]
    return a_opt


def epsilon_greedy(policy, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return policy[s]
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)


def print_values(V, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      v = V.get(GridState(i, j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      a = P.get(GridState(i, j), ' ')
      print("  %s  |" % a, end="")
    print("")
