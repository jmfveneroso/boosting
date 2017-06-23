import sys
import random

# This is just a wrapper class to make it easier to 
# manipulate the Tic Tac Toe dataset.
class TicTacToeDataset:
  def __init__(self, filename):
    txt = open(filename)

    self.folds = 5
    self.training_set = []
    lines = txt.readlines()
    for line in lines:
      case = line.strip().split(',')
      case[9] = (case[9] == 'positive')
      self.training_set.append(case)
    random.shuffle(self.training_set)
    self.fold_size = len(self.training_set) / self.folds

  def get_fold(self, i):
    begin = self.fold_size * i
    end = None if (i == self.folds - 1) else begin + self.fold_size
    return self.training_set[begin : end]

  def get_folds_except(self, fold):
    dataset = []
    for i in range(self.folds):
      if fold == i: continue
      dataset += self.get_fold(i)
    return dataset

if __name__ == '__main__':

  if len(sys.argv) != 2:
    print 'usage: tic_tac_toe_dataset <dataset>'
    sys.exit()

  dataset = TicTacToeDataset(sys.argv[1])
  print "Dataset size:", len(dataset.training_set)
