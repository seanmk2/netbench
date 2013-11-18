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
from sklearn.preprocessing       import scale
from sklearn.decomposition       import PCA
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

    self.in_dim      = in_size
    self.hidden_dim  = hidden_size
    self.out_dim     = out_size

    self.neural_net  = self.create_net(in_size=self.in_dim,hidden_size=self.hidden_dim,out_size=self.out_dim)
    self.k_means_net = None
    self.pca_net     = None

    self.error_pairs = {"k-means":[],"pca":[],"k-pca":[]}



  def create_net(self, in_size, hidden_size, out_size, override=False):
    net             = FeedForwardNetwork()
    in_layer        = LinearLayer(in_size)
    hidden_layer    = TanhLayer(hidden_size)
    out_layer       = LinearLayer(out_size)

    net.addInputModule(in_layer)
    net.addModule(hidden_layer)
    net.addOutputModule(out_layer)

    in_to_hidden    = FullConnection(in_layer, hidden_layer)
    hidden_to_out   = FullConnection(hidden_layer, out_layer)

    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)
    net.sortModules()

    if override == True:
      self.neural_net = net
    else:
      return net



  def create_k_means_net(self, override=True):
    if override == True:
      self.k_means_net = self.create_net(in_size=self.in_dim,hidden_size=self.hidden_dim,out_size=self.out_dim)
    if override == False:
      return self.create_net(in_size=self.in_dim,hidden_size=self.hidden_dim,out_size=self.out_dim)



  def create_pca_net(self, reduced_in_size, override=True):
    if override == True:
      self.pca_net     = self.create_net(in_size=reduced_in_size,hidden_size=self.hidden_dim,out_size=self.out_dim)
    if override == False:
      return self.create_net(in_size=reduced_in_size,hidden_size=self.hidden_dim,out_size=self.out_dim)



  def net_activate(self, data):
    return self.neural_net.activate(data)



  def k_means_net_activate(self, data):
    return self.k_means_net.activate(data)



  def pca_net_activate(self, data):
    return self.pca_net.activate(data)



  def train_many_k_means_reductions(self, dataset, portion=1.00, iters=20, k_means_reductions=[ float(num)/10 for num in xrange(1,10) ]):
    if portion >= 1:
      training_data   = dataset.trn_data
      test_data       = dataset.tst_data
    if portion < 1:
      dataset.get_portion(portion)
      training_data   = dataset.portion["training"]
      test_data       = dataset.portion["test"]
    entro = dataset.entro

    self.iters        = iters
    self.sample_size  = dataset.tot_size

    self.create_net(in_size=self.in_dim,hidden_size=self.hidden_dim,out_size=self.out_dim,override=True)
    neural_trainer = BackpropTrainer(self.neural_net, dataset=training_data, momentum=0.1, verbose=True, weightdecay=0.01)

    for reduction in k_means_reductions:
      dataset.create_k_means_data(k_means_reduction=reduction)
      k_means_training_data = dataset.k_means_training_data

      self.create_k_means_net(override=True)
      k_means_trainer = BackpropTrainer(self.k_means_net, dataset=k_means_training_data, momentum=0.1, verbose=True, weightdecay=0.01)

      full_k_means_pair_errors = []

      for i in range(iters):
        print "Reduction: " + str(float(reduction)*100) + "% of Data, Iteration " + str(i)

        old_stdout   = sys.stdout            ### CAPTURE
        capturer     = StringIO.StringIO()   ### CAPTURE
        sys.stdout   = capturer              ### CAPTURE

        #print "-------------------------"
        neural_trainer.trainEpochs(1)
        #print "---"
        k_means_trainer.trainEpochs(1)

        sys.stdout   = old_stdout            ### CAPTURE
        output       = capturer.getvalue()   ### CAPTURE
        err_pair     = self.process_output_error_pair(output)
        full_k_means_pair_errors.append(err_pair)

      self.error_pairs["k-means"].append(full_k_means_pair_errors)
    self.generate_k_means_error_comparison(k_means_reductions)



  def train_many_pca_reductions(self, dataset, portion=1.00, iters=20, pca_reductions=[ num for num in xrange(3,7) ]):
    if portion >= 1:
      training_data   = dataset.trn_data
      test_data       = dataset.tst_data
    if portion < 1:
      dataset.get_portion(portion)
      training_data   = dataset.portion["training"]
      test_data       = dataset.portion["test"]
    entro = dataset.entro

    self.iters        = iters
    self.sample_size  = dataset.tot_size

    self.create_net(in_size=self.in_dim,hidden_size=self.hidden_dim,out_size=self.out_dim,override=True)
    neural_trainer = BackpropTrainer(self.neural_net, dataset=training_data, momentum=0.1, verbose=True, weightdecay=0.01)

    for reduction in pca_reductions:
      dataset.create_pca_data(pca_dimension_target=reduction)
      pca_training_data = dataset.pca_training_data

      self.create_pca_net(reduction, override=True)
      pca_trainer = BackpropTrainer(self.pca_net, dataset=pca_training_data, momentum=0.1, verbose=True, weightdecay=0.01)


      ###################################
      if reduction == 10:
        for i in xrange(len(pca_training_data['input'])):
          print "____________________________________________"
          print training_data['input'][i]
          print pca_training_data['input'][i]
          print "||||||||||||||||||||||||||||||||||||||||||||"
          print training_data['target'][i]
          print pca_training_data['target'][i]
          ########## USE RMSE AND SEE WHAT HAPPENS
      ###################################

      full_pca_pair_errors = []

      for i in range(iters):
        print "Reduction: " + str(reduction) + " Dimensions of Data, Iteration " + str(i)

        old_stdout   = sys.stdout            ### CAPTURE
        capturer     = StringIO.StringIO()   ### CAPTURE
        sys.stdout   = capturer              ### CAPTURE

        #print "-------------------------"
        neural_trainer.trainEpochs(1)
        #print "---"
        pca_trainer.trainEpochs(1)

        sys.stdout   = old_stdout            ### CAPTURE
        output       = capturer.getvalue()   ### CAPTURE
        err_pair     = self.process_output_error_pair(output)
        full_pca_pair_errors.append(err_pair)

      self.error_pairs["pca"].append(full_pca_pair_errors)
    self.generate_pca_error_comparison(pca_reductions)



  def process_output_error_pair(self, captured_output):
    error_out  = captured_output.split('\n')
    error_out.pop()
    error_pair = [ error_msg.split(': ')[1] for error_msg in error_out ]
    return tuple(error_pair) # pair of (all_data_net error, reduced_data_net error)



  def generate_k_means_error_comparison(self, k_means_reductions):
    ### each pair in list is (full_data error, partial_data error)
    x_i     = [ x for x in xrange(1,len(self.error_pairs["k-means"][0])+1)]
    y_full1 = [ y_pt[0] for y_pt in self.error_pairs["k-means"][0] ]

    plt.hold(True)
    plt.plot(x_i, y_full1, 'k', alpha=1.0, label='1.00')

    alpha_values = [0.20,0.40,0.60,0.80,1.00]

    for i in xrange(len(self.error_pairs["k-means"])):
      y_ = [ y_pt[1] for y_pt in self.error_pairs["k-means"][i] ]
      plt.plot(x_i, y_, 'r', alpha=alpha_values[i], label=str(k_means_reductions[i]))

    plt.legend(loc='upper right')

    plt.ylim(0.10,0.40)
    plt.title("[Total Samples: "+str(self.sample_size)+"] | [Total Iterations: "+str(self.iters)+"]")
    plt.xlabel("[Iteration #]")
    plt.ylabel("[Total Error]")
    plt.show()



  def generate_pca_error_comparison(self, pca_dimension_targets):
    ### each pair in list is (full_data error, partial_data error)
    x_i     = [ x for x in xrange(1,len(self.error_pairs["pca"][0])+1)]
    y_full1 = [ y_pt[0] for y_pt in self.error_pairs["pca"][0] ]

    plt.hold(True)
    plt.plot(x_i, y_full1, 'k', alpha=1.0, label='FULL')

    alpha_values = [0.20,0.40,0.60,0.80,1.00]

    for i in xrange(len(self.error_pairs["pca"])):
      y_ = [ y_pt[1] for y_pt in self.error_pairs["pca"][i] ]
      plt.plot(x_i, y_, 'r', alpha=alpha_values[i], label=str(pca_dimension_targets[i]))

    plt.legend(loc='upper right')

    plt.ylim(0.00,0.40)
    plt.title("[Total Samples: "+str(self.sample_size)+"] | [Total Iterations: "+str(self.iters)+"]")
    plt.xlabel("[Iteration #]")
    plt.ylabel("[Total Error]")
    plt.show()



  # def rmse_evaluation(self, tst_data, entropy):
  #   true_values = tst_data['target']
  #   pred_values = []

  #   for ind in xrange(len(tst_data['target'])):
  #     pred = self.net_activate(tst_data['input'][ind])
  #     pred_values.append(pred)

  #   mse  = mean_squared_error(true_values, pred_values)
  #   rmse = math.sqrt(mse)
  #   normalized_rmse = rmse / entropy

  #   print "       RMSE: " + str(rmse)
  #   print "Normd  RMSE: " + str(normalized_rmse)

  # def k_rmse_evaluation(self, tst_data, entropy):

  #   true_values = tst_data['target']
  #   pred_values = []

  #   for ind in xrange(len(tst_data['target'])):
  #     pred = self.k_net_activate(tst_data['input'][ind])
  #     pred_values.append(pred)

  #   mse  = mean_squared_error(true_values, pred_values)
  #   rmse = math.sqrt(mse)
  #   normalized_rmse = rmse / entropy

  #   print "      KRMSE: " + str(rmse)
  #   print "Normd KRMSE: " + str(normalized_rmse)



class RandomDataSet(object):
  def __init__(self, in_dim, out_dim, size = 1000, means = None, covas = None, split_proportion=0.25):
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

    tst_data, trn_data = all_data.splitWithProportion(split_proportion)

    self.tot_size = size
    self.all_data = all_data
    self.tst_data = tst_data
    self.trn_data = trn_data
    self.portion  = {"training":None,"test":None}

    self.split_proportion = split_proportion



  def get_portion(self, portion=1.00):
    num_portion  = int(self.tot_size * portion)
    data_portion = SupervisedDataSet(self.in_dim, self.out_dim)
    for ind in xrange(num_portion):
      data_portion.addSample(copy.deepcopy(self.all_data['input'][ind]), copy.deepcopy(self.all_data['target'][ind]))

    tst_portion, trn_portion = data_portion.splitWithProportion(self.split_proportion)

    self.portion["training"] = trn_portion
    self.portion["test"]     = tst_portion



  def create_k_means_data(self, k_means_reduction=0.01):
    k_clusters = int(self.tot_size * k_means_reduction * (float(1) - self.split_proportion))

    kmeans = KMeans(n_clusters = k_clusters)
    kmeans.fit(self.trn_data['input'])
    print "fitting data with kmeans..."
    centroids = kmeans.cluster_centers_
    kmeans_trn_data_x = []
    kmeans_trn_data_y = []

    print "finding closest point to each centroid..."
    centroid_count = 0

    indices = []

    for centroid in centroids:

      centroid_count += 1
      if centroid_count % (k_clusters / 5) == 0:
        print "completed "+str(100.0 * float(centroid_count) / float(k_clusters))+"% of search..."

      min_pdist = float("+inf")
      min_index = 0
      for ind in xrange(len(self.trn_data['input'])):
        L2norm = pdist([centroid,self.trn_data['input'][ind]])
        if L2norm < min_pdist:
          min_pdist = L2norm
          min_index = ind

      indices.append(min_index)

      kmeans_trn_data_x.append(copy.deepcopy(self.trn_data['input'][min_index]))
      kmeans_trn_data_y.append(copy.deepcopy(self.trn_data['target'][min_index]))

    print "creating reduced kmeans dataset..."
    kmeans_trn_data = SupervisedDataSet(self.in_dim, self.out_dim)
    for n in xrange(k_clusters):
      kmeans_trn_data.addSample(kmeans_trn_data_x[n], kmeans_trn_data_y[n])

    self.k_means_training_data = kmeans_trn_data



  def create_pca_data(self, pca_dimension_target=2):
    print "creating reduced pca dataset..."
    pca_trn_data_x = PCA(n_components=pca_dimension_target).fit_transform(self.trn_data['input'])
    pca_trn_data_y = self.trn_data['target']
    pca_trn_data   = SupervisedDataSet(pca_dimension_target,self.out_dim)
    for n in xrange(len(pca_trn_data_x)):
      pca_trn_data.addSample(pca_trn_data_x[n], pca_trn_data_y[n])

    self.pca_training_data = pca_trn_data

























