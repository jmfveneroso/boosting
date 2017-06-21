import sys
import time
import random
from numpy import *

class TicTacToeDataset:
  def __init__(self, filename):
    txt = open(filename)

    self.training_cases = []
    self.weights = []
    lines = txt.readlines()
    for line in lines:
      case = line.strip().split(',')
      case[9] = (case[9] == 'positive')
      self.training_cases.append(case)
    print len(self.training_cases)

class AdaBoost:
  def __init__(self, training_set):
    self.training_set = training_set
    self.N = len(self.training_set)
    self.weights = ones(self.N)/self.N
    self.RULES = []
    self.ALPHA = []

  def set_rule(self, func, test=False):
    errors = array([t[1]!=func(t[0]) for t in self.training_set])
    e = (errors*self.weights).sum()
    if test: return e
    alpha = 0.5 * log((1-e)/e)
    print 'e=%.2f a=%.2f'%(e, alpha)
    w = zeros(self.N)
    for i in range(self.N):
        if errors[i] == 1: w[i] = self.weights[i] * exp(alpha)
        else: w[i] = self.weights[i] * exp(-alpha)
    self.weights = w / w.sum()
    self.RULES.append(func)
    self.ALPHA.append(alpha)

  def evaluate(self):
    NR = len(self.RULES)
    for (x,l) in self.training_set:
        hx = [self.ALPHA[i]*self.RULES[i](x) for i in range(NR)]
        print x, sign(l) == sign(sum(hx))

if __name__ == '__main__':

  if len(sys.argv) != 2:
    print 'usage: boosting <dataset>'
    sys.exit()

  t0 = time.time()
  dataset = TicTacToeDataset(sys.argv[1])
  ms = (time.time() - t0) * 1000
  print "ms:", ms
