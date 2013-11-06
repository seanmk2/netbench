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

import sys
import copy
import math
import random
import StringIO
import matplotlib.pyplot as plt


class NeuralNet(object):
  def __init__(self, in_size, hidden_size, out_size):
    net             = FeedForwardNetwork()
    in_layer        = LinearLayer(in_size)
    hidden_layer    = TanhLayer(hidden_size)
    out_layer       = LinearLayer(out_size)

    k_net           = FeedForwardNetwork()
    k_in_layer      = LinearLayer(in_size)
    k_hidden_layer  = TanhLayer(hidden_size)
    k_out_layer     = LinearLayer(out_size)

    print "creating full data neural net..."
    net.addInputModule(in_layer)
    net.addModule(hidden_layer)
    net.addOutputModule(out_layer)

    k_net.addInputModule(k_in_layer)
    k_net.addModule(k_hidden_layer)
    k_net.addOutputModule(k_out_layer)

    in_to_hidden  = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)

    k_in_to_hidden  = FullConnection(k_in_layer, k_hidden_layer)
    k_hidden_to_out = FullConnection(k_hidden_layer, k_out_layer)

    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)

    k_net.addConnection(k_in_to_hidden)
    k_net.addConnection(k_hidden_to_out)

    print "creating partial data neural net..."
    net.sortModules()
    k_net.sortModules()

    self.neural_net   = net
    self.k_neural_net = k_net
    self.in_dim       = in_size
    self.hidden_dim   = hidden_size
    self.out_dim      = out_size
    self.error_pairs  = []

  def create_another_net(self, override=False):
    another_net     = FeedForwardNetwork()
    in_layer        = LinearLayer(self.in_dim)
    hidden_layer    = TanhLayer(self.hidden_dim)
    out_layer       = LinearLayer(self.out_dim)

    another_net.addInputModule(in_layer)
    another_net.addModule(hidden_layer)
    another_net.addOutputModule(out_layer)

    in_to_hidden  = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)

    another_net.addConnection(in_to_hidden)
    another_net.addConnection(hidden_to_out)

    another_net.sortModules()

    if override == True:
      self.neural_net  = another_net
    else:
      self.another_net = another_net

  def net_activate(self, data):
    return self.neural_net.activate(data)

  def k_net_activate(self, data):
    return self.k_neural_net.activate(data)

  def net_params(self):
    print self.neural_net.params

  def k_net_params(self):
    print self.k_neural_net.params

  def train_many_k_reductions(self, size=10000, iters=20, k_reductions=[ float(num)/20 for num in xrange(1,20) ]):
    random_data_set = RandomDataSet(self.in_dim, self.out_dim, size)
    all_data        = random_data_set.all_data
    trn_data        = random_data_set.trn_data
    tst_data        = random_data_set.tst_data
    entro           = random_data_set.entro



    for reduction in k_reductions:
      random_data_set.create_kmeans_reduced_trn_data(k_reduction = reduction)
      k_trn_data = random_data_set.kmeans_trn_data

      self.create_another_net(override=True)
      trainer = BackpropTrainer(self.neural_net, dataset=trn_data, momentum=0.1, verbose=True, weightdecay=0.01)

      self.create_another_net()
      reduction_trainer = BackpropTrainer(self.another_net, dataset=k_trn_data, momentum=0.1, verbose=True, weightdecay=0.01)

      full_reduced_pair_errors = []

      for i in range(iters):
        print "Reduction: " + str(float(reduction)*100) + "% of Data, Iteration " + str(i)

        old_stdout   = sys.stdout            ### CAPTURE
        capturer     = StringIO.StringIO()   ### CAPTURE
        sys.stdout   = capturer              ### CAPTURE

        #print "-------------------------"
        trainer.trainEpochs(1)
        #print "---"
        reduction_trainer.trainEpochs(1)

        sys.stdout   = old_stdout            ### CAPTURE
        output       = capturer.getvalue()   ### CAPTURE
        err_pair     = self.process_output_error_pair(output)
        full_reduced_pair_errors.append(err_pair)

      self.error_pairs.append(full_reduced_pair_errors)

    self.generate_full_reduced_error_comparison(k_reductions)

  def process_output_error_pair(self, captured_output):
    error_out  = captured_output.split('\n')
    error_out.pop()
    error_pair = [ error_msg.split(': ')[1] for error_msg in error_out ]
    return tuple(error_pair) # pair of (all_data_net error, reduced_data_net error)

  def generate_full_reduced_error_comparison(self, performed_reductions):
    ### Verify that errors generated by full model are about the same for each pair
    ### performed reductions lets us label graph with knowledge of data reduction
    ### each pair in list is (full_data error, partial_data error)
    ### Matplotlib graph generation

    try:
      x_i    = [ x for x in xrange(1,len(self.error_pairs[0])+1)]
      y_full1 = [ y_pt[0] for y_pt in self.error_pairs[0] ]
      #y_full2 = [ y_pt[0] for y_pt in self.error_pairs[1] ]
      #y_full2 = [ y_pt[0] for y_pt in self.error_pairs[2] ]

      y_1    = [ y_pt[1] for y_pt in self.error_pairs[0] ]
      y_2    = [ y_pt[1] for y_pt in self.error_pairs[1] ]
      y_3    = [ y_pt[1] for y_pt in self.error_pairs[2] ]

      plt.hold(True)
      plt.plot(x_i, y_full1, 'k', alpha=1.0)
      #plt.plot(x_i, y_full2, 'k', alpha=1.0)
      #plt.plot(x_i, y_full3, 'k', alpha=1.0)
      plt.plot(x_i, y_1,    'r', alpha=.25)
      plt.plot(x_i, y_2,    'r', alpha=.50)
      plt.plot(x_i, y_3,    'r', alpha=.75)

      plt.show()
    except:
      return 0
    return 0

  def train_random(self, size=10000, iters=20, reduction=0.10):
    random_data_set = RandomDataSet(self.in_dim, self.out_dim, size)
    all_data        = random_data_set.all_data
    trn_data        = random_data_set.trn_data
    tst_data        = random_data_set.tst_data
    entro           = random_data_set.entro

    random_data_set.create_kmeans_reduced_trn_data(k_reduction = reduction)
    k_trn_data      = random_data_set.kmeans_trn_data

    print "training on complete data set..."
    trainer = BackpropTrainer(self.neural_net, dataset=trn_data, momentum=0.1, verbose=True, weightdecay=0.01)

    print "training on kmeans reduced data set..."
    k_trainer = BackpropTrainer(self.k_neural_net, dataset=k_trn_data, momentum=0.1, verbose=True, weightdecay=0.01)

    for i in range(iters):

      print "-------------------------"
      trainer.trainEpochs(1)
      print "---"
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

    print "       RMSE: " + str(rmse)
    print "Normd  RMSE: " + str(normalized_rmse)

  def k_rmse_evaluation(self, tst_data, entropy):

    true_values = tst_data['target']
    pred_values = []

    for ind in xrange(len(tst_data['target'])):
      pred = self.k_net_activate(tst_data['input'][ind])
      pred_values.append(pred)

    mse  = mean_squared_error(true_values, pred_values)
    rmse = math.sqrt(mse)
    normalized_rmse = rmse / entropy

    print "      KRMSE: " + str(rmse)
    print "Normd KRMSE: " + str(normalized_rmse)

class RandomDataSet(object):
  def __init__(self, in_dim, out_dim, size = 1000, means = None, covas = None):
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
  def create_kmeans_reduced_trn_data(self, k_reduction = 0.01):
    k_clusters = int(self.tot_size * k_reduction * (float(1) - self.split_proportion))

    kmeans = KMeans(n_clusters = k_clusters)
    kmeans.fit(self.trn_data['input'])
    print "fitting data with kmeans..."
    centroids = kmeans.cluster_centers_
    kmeans_trn_data_x = []
    kmeans_trn_data_y = []

    print "finding closest point to each centroid..."
    centroid_count = 0

    ###
    #print "len(centroids) = " + str(len(centroids))
    indices = []
    ###

    for centroid in centroids:

      ###
      #print "printing centroid: "
      #print centroid
      ###

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

      ###
      #print "printing index: "
      #print min_index
      ###
      ###
      # print "      ind was: " + str(ind)
      # print "min_index was: " + str(min_index)
      indices.append(min_index)
      ###

      kmeans_trn_data_x.append(copy.deepcopy(self.trn_data['input'][min_index]))
      kmeans_trn_data_y.append(copy.deepcopy(self.trn_data['target'][min_index]))

    ###
    #print "len(kmeans_trn_data_x) = " + str(len(kmeans_trn_data_x))
    #print "len(kmeans_trn_data_y) = " + str(len(kmeans_trn_data_y))
    # print sorted(indices)
    ###

    print "creating reduced kmeans dataset..."
    kmeans_trn_data = SupervisedDataSet(self.in_dim, self.out_dim)
    for n in xrange(k_clusters):
      kmeans_trn_data.addSample(kmeans_trn_data_x[n], kmeans_trn_data_y[n])

    self.kmeans_trn_data = kmeans_trn_data
























