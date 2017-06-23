import sys
import time
import random
import numpy as np
import tic_tac_toe_dataset as ds

class DecisionTree:
  def __init__(self, dataset, max_depth):
    self.dataset = dataset
    self.max_depth = max_depth
    self.root = None

  # The Gini index measures the probability of a misclassification.
  def gini_index(self, groups):
    gini = 0.0
    for g in groups:
      # for class_value in (True, False):
      if len(g) == 0: 
        gini += 1
        continue

      total_sum, true_sum, false_sum = 0, 0, 0      
      for row in g:
        if row[-2] == True: true_sum += row[-1]
        if row[-2] == False: false_sum += row[-1]
        total_sum += row[-1]

      probability_true = true_sum / total_sum
      probability_false = false_sum / total_sum
      gini += probability_true * (1 - probability_true)
      gini += probability_false * (1 - probability_false)
      # probability = [row[-2] for row in g].count(class_value) / float(len(g))
      # gini += probability * (1 - probability)
    return gini

  # Performs a split in a dataset according to the value of a column.
  def test_split(self, index, value, dataset):
    left, right = list(), list()
    for row in dataset:
      if row[index] == value: 
        left.append(row)
      else: 
        right.append(row)
    return left, right

  # Gets the best split possible based on the Gini index.
  def get_best_split(self, dataset):
    best_index, best_value, best_score = 999, 999, 999
    best_groups = None
    for index in range(len(dataset[0]) - 2):
      for value in ['x', 'o', 'b']:
        groups = self.test_split(index, value, dataset)
        gini = self.gini_index(groups)
        if gini < best_score:
          best_index, best_value, best_score = index, value, gini
          best_groups = groups
    return { 'index': best_index, 'value': best_value, 
             'groups': best_groups }

  # Creates a leaf node.
  def to_terminal(self, group):
    probability_true = 0
    probability_false = 0
    total_sum = 0
    for row in group:
      if row[-2] == True: probability_true += row[-1]
      else: probability_false += row[-1]
      total_sum = row[-1]

    probability_true /= total_sum
    probability_false /= total_sum

    return probability_true > probability_false
    # outcomes = [row[-2] for row in group]
    # print outcomes.count(True), outcomes.count(False)
    # print outcomes
    # return max(set(outcomes), key = outcomes.count)

  # Recursive split procedure.
  def split(self, node, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
      node['left'] = node['right'] = self.to_terminal(left + right)
      return

    if depth >= self.max_depth:
      node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
      return

    # Process left child.
    node['left'] = self.get_best_split(left)
    self.split(node['left'], depth + 1)

    # Process right child.
    node['right'] = self.get_best_split(right)
    self.split(node['right'], depth + 1)

  # Builds the decision tree according to the provided dataset.
  def build(self):
    self.root = self.get_best_split(self.dataset)
    self.split(self.root, 1)

  # Predicts an outcome based for a row.
  def predict(self, row, node = None):
    if node == None:
      return self.predict(row, self.root)

    if row[node['index']] == node['value']:
      if isinstance(node['left'], dict):
        return self.predict(row, node['left'])
      else:
        return node['left']
    else:
      if isinstance(node['right'], dict):
        return self.predict(row, node['right'])
      else:
        return node['right']

  # Prints the decision tree.
  def print_tree(self, node = None, depth = 0):
    if node == None:
      return self.print_tree(self.root, depth)

    if isinstance(node, dict):
      print('%s[X[%d] = %s]' % ((depth * ' ', (node['index'] + 1), node['value'])))
      self.print_tree(node['left'], depth + 1)
      self.print_tree(node['right'], depth + 1)
    else:
      print('%s[%s]' % ((depth * ' ', node)))

# Trains a decision tree from scratch using the training set and 
# validates the accuracy with the validation set.
def fit_tree(n, training_set, validation_set):
  decision_tree = DecisionTree(training_set, n)
  decision_tree.build()

  training_error = 0.0
  for test in training_set:
    result = decision_tree.predict(test)
    if result != test[-2]:
       training_error += 1
  training_error /= len(training_set)

  validation_error = 0.0
  for test in validation_set:
    result = decision_tree.predict(test)
    if result != test[-2]:
       validation_error += 1
  validation_error /= len(validation_set)

  return training_error, validation_error

# Uses cross validation to measure the efficiency of an n-depth tree.
def cross_validation(n, dataset):
  training_error, validation_error = 0, 0
  for fold in range(5):
    training_set = dataset.get_fold(fold)
    validation_set = dataset.get_folds_except(fold)
    t_error, v_error = fit_tree(n, training_set, validation_set)
    training_error += t_error
    validation_error += v_error

  training_error /= 5
  validation_error /= 5
  print "n:", n, "training error:", training_error, "validation error:", validation_error

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'usage: decision_tree <dataset>'
    sys.exit()

  t0 = time.time()
  dataset = ds.TicTacToeDataset(sys.argv[1])
  for row in dataset.training_set:
    row.append(1)

  for n in range(1, 40):
    cross_validation(n, dataset)

  ms = (time.time() - t0) * 1000
  print "ms:", ms
