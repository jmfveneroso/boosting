import sys
import time
import random
import numpy as np
from math import log, exp
from tic_tac_toe_dataset import TicTacToeDataset
from decision_tree import DecisionTree

class AdaBoost:
  def __init__(self, dataset, depth):
    self.dataset = dataset
    self.weak_learners = []
    self.alpha = []
    self.depth = depth

    # The weight of a row in the training set is acually the
    # last value in that row.
    for row in self.dataset:
      row[-1] = 1 / float(len(self.dataset))
 
  # Runs an Ada Boost iteration by creating a new 
  # weak learner and calculating its alpha and 
  # updating the training set weights.
  def run_iteration(self):
    weak_learner = DecisionTree(self.dataset, self.depth)
    weak_learner.build()
   
    e = 0 
    errors = []
    for row in self.dataset:
      if weak_learner.predict(row) != row[-2]:
        e += row[-1]
        errors.append(1)
      else:
        errors.append(0)

    alpha = 0.5 * log((1 - e) / e)
    # print 'e=%.2f a=%.2f'%(e, alpha)

    sum_weights = 0
    for i in range(len(self.dataset)):
      row = self.dataset[i]
      if errors[i] == 1: row[-1] = row[-1] * exp(alpha)
      else: row[-1] = row[-1] * exp(-alpha)
      sum_weights += row[-1]

    for row in self.dataset:
      row[-1] /= sum_weights

    self.weak_learners.append(weak_learner)
    self.alpha.append(alpha)

  # Makes a prediction for a row based on all the weak learners
  # created.
  def predict(self, row):
    n = len(self.weak_learners)
    hx = 0
    for i in range(n):
      if self.weak_learners[i].predict(row):
        hx += self.alpha[i]
      else:
        hx -= self.alpha[i]

    return hx > 0

# Creates an Ada Boost learner with n iteration and performs
# cross validation.
def fit(n, depth, training_set, validation_set):
  adaboost = AdaBoost(training_set, depth)
  for n in range(n):
    adaboost.run_iteration()

  training_error = 0.0
  for test in training_set:
    result = adaboost.predict(test)
    if result != test[-2]:
       training_error += 1
  training_error /= len(training_set)

  validation_error = 0.0
  for test in validation_set:
    result = adaboost.predict(test)
    if result != test[-2]:
       validation_error += 1
  validation_error /= len(validation_set)

  return training_error, validation_error

# Performs all fold permutations and prints the result of 
# cross validation for n iterations of the Ada Boost algorithm.
def cross_validation(n, depth, dataset):
  training_error, validation_error = 0, 0
  for fold in range(5):
    training_set = dataset.get_fold(fold)
    validation_set = dataset.get_folds_except(fold)
    t_error, v_error = fit(n, depth, training_set, validation_set)
    training_error += t_error
    validation_error += v_error

  training_error /= 5
  validation_error /= 5
  print "n:", n, "training error:", training_error, "validation error:", validation_error

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'usage: boosting <dataset>'
    sys.exit()

  t0 = time.time()
  dataset = TicTacToeDataset(sys.argv[1])
  for row in dataset.training_set:
    row.append(1)

  for n in range(1, 400):
    cross_validation(n, 1, dataset)

  ms = (time.time() - t0) * 1000
  print "ms:", ms
