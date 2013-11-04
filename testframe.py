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
from sklearn.cluster             import KMeans
from scipy                       import diag, arange, meshgrid, where
from scipy.spatial.distance      import pdist

import copy
import math
import random

class NeuralNet(object):
  def __init__(self, in_size, hidden_size, out_size):
    net           = FeedForwardNetwork()
    in_layer      = LinearLayer(in_size)
    hidden_layer  = TanhLayer(hidden_size)
    out_layer     = LinearLayer(out_size)

    print "creating full data neural net..."
    net.addInputModule(in_layer)
    net.addModule(hidden_layer)
    net.addOutputModule(out_layer)

    in_to_hidden  = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)

    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)

    print "creating partial data neural net..."
    net.sortModules()
    self.neural_net   = net
    self.k_neural_net = copy.deepcopy(net)
    self.in_dim       = in_size
    self.out_dim      = out_size

  def net_activate(self, data):
    return self.neural_net.activate(data)

  def k_net_activate(self, data):
    return self.k_neural_net.activate(data)

  def net_params(self):
    print self.neural_net.params

  def k_net_params(self):
    print self.k_neural_net.params

  def train_random(self, size = 10000, iters=20):
    random_data_set = RandomDataSet(self.in_dim, self.out_dim, size)
    all_data        = random_data_set.all_data
    trn_data        = random_data_set.trn_data
    tst_data        = random_data_set.tst_data
    entro           = random_data_set.entro

    random_data_set.create_kmeans_reduced_trn_data()
    k_trn_data      = random_data_set.kmeans_trn_data

    print "training on complete data set..."
    trainer = BackpropTrainer(self.neural_net, dataset=trn_data, momentum=0.1, verbose=True, weightdecay=0.01)
    print "training on kmeans reduced data set..."
    k_trainer = BackpropTrainer(self.k_neural_net, dataset=k_trn_data, momentum=0.1, verbose=True, weightdecay=0.01)

    for i in range(iters):
      trainer.trainEpochs(1)
      k_trainer.trainEpochs(1)

      self.rmse_evaluation(tst_data, entro)
      self.k_rmse_evaluation(tst_data, entro)

    self.trainer   = trainer
    self.k_trainer = k_trainer

  def rmse_evaluation(self, tst_data, entropy):
    true_values = tst_data['target']
    pred_values = []

    for ind in xrange(len(tst_data['target'])):
      pred = self.net_activate(tst_data['input'][ind])
      pred_values.append(pred)

    mse  = mean_squared_error(true_values, pred_values)
    rmse = math.sqrt(mse)
    normalized_rmse = rmse / entropy

    print "            RMSE: " + str(rmse)
    print "Normalized  RMSE: " + str(normalized_rmse)

  def k_rmse_evaluation(self, tst_data, entropy):

    true_values = tst_data['target']
    pred_values = []

    for ind in xrange(len(tst_data['target'])):
      pred = self.k_net_activate(tst_data['input'][ind])
      pred_values.append(pred)

    mse  = mean_squared_error(true_values, pred_values)
    rmse = math.sqrt(mse)
    normalized_rmse = rmse / entropy

    print "           KRMSE: " + str(rmse)
    print "Normalized KRMSE: " + str(normalized_rmse)

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

    self.in_dim  = in_dim
    self.out_dim = out_dim

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


    split_proportion   = 0.25
    tst_data, trn_data = all_data.splitWithProportion(split_proportion)

    self.tot_size = size
    self.all_data = all_data
    self.tst_data = tst_data
    self.trn_data = trn_data

    self.split_proportion = split_proportion



  #total error is the same each time, perhaps need to do this step before creating SupervisedDataSet
  def create_kmeans_reduced_trn_data(self, k_reduction = 0.80):
    k_clusters = int(self.tot_size * k_reduction * (float(1) - self.split_proportion))

    kmeans = KMeans(n_clusters = k_clusters)
    kmeans.fit(self.trn_data['input'])
    print "fitting data with kmeans..."
    centroids = kmeans.cluster_centers_
    kmeans_trn_data_x = []
    kmeans_trn_data_y = []

    print "finding closest point to each centroid..."
    centroid_count = 0
    for centroid in centroids:

      centroid_count += 1
      if centroid_count % (k_clusters / 20) == 0:
        print "completed "+str(100.0 * float(centroid_count) / float(k_clusters))+"% of search..."

      min_pdist = float("+inf")
      min_index = 0
      for ind in xrange(len(self.trn_data['input'])):
        L2norm = pdist([centroid,self.trn_data['input'][ind]])
        if L2norm < min_pdist:
          min_pdist = L2norm
          min_index = ind
      kmeans_trn_data_x.append(copy.deepcopy(self.trn_data['input'][ind]))
      kmeans_trn_data_y.append(copy.deepcopy(self.trn_data['target'][ind]))

    print "creating reduced kmeans dataset..."
    kmeans_trn_data = SupervisedDataSet(self.in_dim, self.out_dim)
    for n in xrange(k_clusters):
      kmeans_trn_data.addSample(kmeans_trn_data_x[n], kmeans_trn_data_y[n])

    self.kmeans_trn_data = kmeans_trn_data
























