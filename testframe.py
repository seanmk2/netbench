from pybrain.structure           import FeedForwardNetwork
from pybrain.structure           import LinearLayer
from pybrain.structure           import SigmoidLayer
from pybrain.structure           import TanhLayer
from pybrain.structure           import FullConnection
from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.supervised.trainers import BackpropTrainer
from numpy.random                import multivariate_normal
from numpy.linalg                import det
from sklearn.metrics             import mean_squared_error
from scipy                       import diag, arange, meshgrid, where

import math
import random

class NeuralNet(object):
  def __init__(self, in_size, hidden_size, out_size):
    net           = FeedForwardNetwork()
    in_layer      = LinearLayer(in_size)
    hidden_layer  = TanhLayer(hidden_size)
    out_layer     = LinearLayer(out_size)

    net.addInputModule(in_layer)
    net.addModule(hidden_layer)
    net.addOutputModule(out_layer)

    in_to_hidden  = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)

    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)

    net.sortModules()
    self.neural_net = net
    self.in_dim     = in_size
    self.out_dim    = out_size

  def net_activate(self, data):
    return self.neural_net.activate(data)

  def net_params(self):
    print self.neural_net.params

  def train_random(self, size = 10000, iters=20):
    random_data_set = RandomDataSet(self.in_dim, self.out_dim, size)
    all_data        = random_data_set.all_data
    trn_data        = random_data_set.trn_data
    tst_data        = random_data_set.tst_data
    entro           = random_data_set.entro

    trainer = BackpropTrainer(self.neural_net, dataset=trn_data, momentum=0.1, verbose=True, weightdecay=0.01)

    for i in range(iters):
      trainer.trainEpochs(5)
      self.rmse_evaluation(tst_data, entro)

    self.trainer = trainer

  def rmse_evaluation(self, tst_data, entropy):
    true_values = tst_data['target']
    pred_values = []

    for ind in xrange(len(tst_data['target'])):
      pred = self.net_activate(tst_data['input'][ind])
      pred_values.append(pred)

    mse  = mean_squared_error(true_values, pred_values)
    rmse = math.sqrt(mse)
    normalized_rmse = rmse / entropy

    print "RMSE: " + str(rmse)
    print "Normalized RMSE: " + str(normalized_rmse)

class RandomDataSet(object):
  def __init__(self, in_dim, out_dim, size = 10000, means = None, covas = None):
    if means == None or covas == None:
      means = []
      covas = []

      for i in xrange(in_dim):

        ### randomMeans
        sign_value = random.choice([-1,1])
        means.append(random.random()*10*sign_value)

        ### randomCovas
        size_value = 3
        covas.append(random.random()*size_value)

    means = tuple(means)
    covas = diag(covas)
    entro = math.log(det(covas))

    self.means = means
    self.covas = covas
    self.entro = entro

    all_data = SupervisedDataSet(in_dim, out_dim)
    for n in xrange(size):
      in_datum  = multivariate_normal(means,covas)
      out_datum = []
      for z in xrange(out_dim):
        start_ind = z * (in_dim / out_dim)
        end_ind   = (z + 1) * (in_dim / out_dim) + 1
        val       = math.sin(sum(in_datum[start_ind:end_ind]))
        out_datum.append(val)
      all_data.addSample(in_datum,out_datum)

    tst_data, trn_data = all_data.splitWithProportion(0.25)

    self.all_data = all_data
    self.tst_data = tst_data
    self.trn_data = trn_data






# class DataSet(object):
#   def __init__(self, in_data, out_data):
#     train_set = SupervisedDataSet(len(in_data[0]), len(out_data[0]))
#     test_set  = SupervisedDataSet(len(in_data[0]), len(out_data[0]))
#     for ind in xrange(len(in_data)):
#       if ind % 5 == 0:
#         test_set.appendLinked(in_data[ind], out_data[ind])
#       else:
#         train_set.appendLinked(in_data[ind], out_data[ind])

#     self.train_set = train_set
#     self.test_set  = test_set























