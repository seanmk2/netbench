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
from scipy                       import stats
from scipy.spatial.distance      import pdist
from collections                 import Counter

import re
import sys
import copy
import math
import random
import StringIO
import matplotlib.pyplot as plt



# TODO: explore use of neurolab if pybrain does not yield good results



class NeuralNet(object):
  def __init__(self, in_size, hidden_size, out_size):

    self.in_dim      = in_size
    self.hidden_dim  = hidden_size
    self.out_dim     = out_size

    self.neural_net  = self.create_net(in_size=self.in_dim,hidden_size=self.hidden_dim,out_size=self.out_dim)
    self.k_means_net = None
    self.pca_net     = None

    self.trn_error_pairs = {"k-means":[],"pca":[],"k-pca":[]}
    self.tst_error_pairs = {"k-means":[],"pca":[],"k-pca":[]}



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



  def keep_data_within_bounds(self, training_data, lower, upper):
    # TODO: make sure this is working
    if len(training_data['target'][0]) > 1:
      print "Output dimension greater than 1, returning data unmodified (intended only for 1-D output)."
      return training_data
    else:
      in_dim    = len(training_data['input'][0])
      out_dim   = len(training_data['target'][0])
      new_train_x_data = []
      new_train_y_data = []
      discarded_data = 0
      total_data = len(training_data['target'])
      for ind in xrange(len(training_data['target'])):
        if training_data['target'][ind] > lower and training_data['target'][ind] < upper:
          new_train_x_data.append(training_data['input'][ind])
          new_train_y_data.append(training_data['target'][ind])
        else:
          discarded_data += 1

      training_data = SupervisedDataSet(in_dim, out_dim)
      for n in xrange(len(new_train_x_data)):
        training_data.addSample(new_train_x_data[n], new_train_y_data[n])
      print str(discarded_data) + " number of samples discarded out of " + str(total_data)
      return training_data


  def train_many_k_means_reductions(self, dataset, portion=1.00, iters=20, k_means_reductions=[ float(num)/10 for num in xrange(1,10) ], outlier_cutoff=0):
    if portion >= 1:
      training_data   = copy.deepcopy(dataset.trn_data)
      test_data       = copy.deepcopy(dataset.tst_data)
    if portion < 1:
      dataset.get_portion(portion)
      training_data   = copy.deepcopy(dataset.portion["training"])
      test_data       = copy.deepcopy(dataset.portion["test"])
    entro = dataset.entro

    if outlier_cutoff > 0.0:
      low_bound = stats.scoreatpercentile(training_data['target'], outlier_cutoff)
      up_bound  = stats.scoreatpercentile(training_data['target'], 100 - outlier_cutoff)
      training_data = self.keep_data_within_bounds(training_data, low_bound, up_bound)

    self.iters        = iters
    self.sample_size  = dataset.tot_size

    self.create_net(in_size=self.in_dim,hidden_size=self.hidden_dim,out_size=self.out_dim,override=True)
    neural_trainer = BackpropTrainer(self.neural_net, dataset=training_data, momentum=0.1, verbose=True, weightdecay=0.01)

    for reduction in k_means_reductions:
      dataset.create_k_means_data(k_means_reduction=reduction)
      k_means_training_data = dataset.k_means_training_data

      self.create_k_means_net(override=True)
      k_means_trainer = BackpropTrainer(self.k_means_net, dataset=k_means_training_data, momentum=0.1, verbose=True, weightdecay=0.01)

      trn_k_means_pair_errors = []
      tst_k_means_pair_errors = []

      for i in range(iters):
        print "Reduction: " + str(float(reduction)*100) + "% of Data, Iteration " + str(i)

        # old_stdout   = sys.stdout            ### CAPTURE
        # capturer     = StringIO.StringIO()   ### CAPTURE
        # sys.stdout   = capturer              ### CAPTURE

        #print "-------------------------"
        neural_trainer.trainEpochs(1)
        #print "---"
        k_means_trainer.trainEpochs(1)

        # sys.stdout   = old_stdout            ### CAPTURE
        # output       = capturer.getvalue()   ### CAPTURE
        # trn_err_pair = self.process_output_error_pair(output)

        trn_err_pair = []
        trn_err_pair.append(self.nrmsd_evaluation(training_data,"full"))
        trn_err_pair.append(self.nrmsd_evaluation(k_means_training_data,"k-means"))

        trn_k_means_pair_errors.append(tuple(trn_err_pair))

      self.trn_error_pairs["k-means"].append(trn_k_means_pair_errors)
    self.generate_k_means_error_comparison(k_means_reductions)



  def train_many_pca_reductions(self, dataset, portion=1.00, iters=20, pca_reductions=[ num for num in xrange(3,7) ], outlier_cutoff=0):
    if portion >= 1:
      training_data   = copy.deepcopy(dataset.trn_data)
      test_data       = copy.deepcopy(dataset.tst_data)
    if portion < 1:
      dataset.get_portion(portion)
      training_data   = copy.deepcopy(dataset.portion["training"])
      test_data       = copy.deepcopy(dataset.portion["test"])
    entro = dataset.entro

    if outlier_cutoff > 0.0:
      low_bound = stats.scoreatpercentile(training_data['target'], outlier_cutoff)
      up_bound  = stats.scoreatpercentile(training_data['target'], 100 - outlier_cutoff)
      training_data = self.keep_data_within_bounds(training_data, low_bound, up_bound)

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

      trn_pca_pair_errors = []
      tst_pca_pair_errors = []

      for i in range(iters):
        print "Reduction: " + str(reduction) + " Dimensions of Data, Iteration " + str(i)

        # old_stdout   = sys.stdout            ### CAPTURE
        # capturer     = StringIO.StringIO()   ### CAPTURE
        # sys.stdout   = capturer              ### CAPTURE

        #print "-------------------------"
        neural_trainer.trainEpochs(1)
        #print "---"
        pca_trainer.trainEpochs(1)

        # sys.stdout   = old_stdout            ### CAPTURE
        # output       = capturer.getvalue()   ### CAPTURE
        # trn_err_pair = self.process_output_error_pair(output)

        trn_err_pair = []
        trn_err_pair.append(self.nrmsd_evaluation(training_data,"full"))
        trn_err_pair.append(self.nrmsd_evaluation(pca_training_data,"pca"))

        trn_pca_pair_errors.append(tuple(trn_err_pair))

      self.trn_error_pairs["pca"].append(trn_pca_pair_errors)
    self.generate_pca_error_comparison(pca_reductions)



  def process_output_error_pair(self, captured_output):
    error_out  = captured_output.split('\n')
    error_out.pop()
    error_pair = [ error_msg.split(': ')[1] for error_msg in error_out ]
    return tuple(error_pair) # pair of (all_data_net error, reduced_data_net error)



  def generate_k_means_error_comparison(self, k_means_reductions):
    ### each pair in list is (full_data error, partial_data error)
    # TODO: graph training and testing error on same plot, dashed vs. solid lines
    x_i     = [ x for x in xrange(1,len(self.trn_error_pairs["k-means"][0])+1)]
    y_full1 = [ y_pt[0] for y_pt in self.trn_error_pairs["k-means"][0] ]

    plt.hold(True)
    plt.plot(x_i, y_full1, 'k', alpha=1.0, label='1.00')

    alpha_values = [0.20,0.40,0.60,0.80,1.00]

    for i in xrange(len(self.trn_error_pairs["k-means"])):
      print i ##################
      # TODO: fix indexing error
      y_ = [ y_pt[1] for y_pt in self.trn_error_pairs["k-means"][i] ]
      plt.plot(x_i, y_, 'r', alpha=alpha_values[i], label=str(k_means_reductions[i]))

    plt.legend(loc='upper right')

    # TODO: adaptive ylim for error
    #plt.ylim(0.10,0.40)
    plt.title("[Total Samples: "+str(self.sample_size)+"] | [Total Iterations: "+str(self.iters)+"]")
    plt.xlabel("[Iteration #]")
    plt.ylabel("[Error]")
    plt.show()



  def generate_pca_error_comparison(self, pca_dimension_targets):
    ### each pair in list is (full_data error, partial_data error)
    # TODO: graph training and testing error on same plot, dashed vs. solid lines
    x_i     = [ x for x in xrange(1,len(self.trn_error_pairs["pca"][0])+1)]
    y_full1 = [ y_pt[0] for y_pt in self.trn_error_pairs["pca"][0] ]

    plt.hold(True)
    plt.plot(x_i, y_full1, 'k', alpha=1.0, label='FULL')

    alpha_values = [0.20,0.40,0.60,0.80,1.00]

    for i in xrange(len(self.trn_error_pairs["pca"])):
      y_ = [ y_pt[1] for y_pt in self.trn_error_pairs["pca"][i] ]
      plt.plot(x_i, y_, 'r', alpha=alpha_values[i], label=str(pca_dimension_targets[i]))

    plt.legend(loc='upper right')

    # TODO: adaptive ylim for error
    #plt.ylim(0.00,1.00)
    plt.title("[Total Samples: "+str(self.sample_size)+"] | [Total Iterations: "+str(self.iters)+"]")
    plt.xlabel("[Iteration #]")
    plt.ylabel("[Error]")
    plt.show()



  def graph_pca_reduced_data_heatmap(self, target_data):
    # TODO: implement this to graph reduced data
    return None



  def nrmsd_evaluation(self, data, net_type):
    # TODO: explore some actual predicted outputs from given inputs, sanity check to make sure it is "close"
    true_values = data['target']
    pred_values = []

    data_len    = len(data['target'])

    for ind in xrange(data_len):

      if   net_type == "k-means":
        pred = self.k_means_net_activate(data['input'][ind])
      elif net_type == "pca":
        pred = self.pca_net_activate(data['input'][ind])
      else:
        pred = self.net_activate(data['input'][ind])

      pred_values.append(pred)

    msd   = mean_squared_error(true_values, pred_values)
    rmsd  = math.sqrt(msd)
    nrmsd = float(rmsd) / data_len
    return nrmsd



class RandomDataSet(object):
  # TODO: reorganize to dataset class, create random or extract real data subfunctions
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

    # TODO: investigate what the output function actually looks like
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

    # TODO: analyze to see if it matters to use a weighted k means sampling type algorithm or not
    print kmeans.cluster_centers_ ##################################
    print kmeans.labels_ ###########################################

    # TODO: graph weighted k means count dict as histogram to see if distribution matters
    count_dict = Counter()
    total_coun = 0
    for val in kmeans.labels_:
      count_dict[val] += 1
      total_coun += 1

    print count_dict ###############################################
    print total_coun ###############################################

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

  def extract_turbulence_data(self, target_file_name, desired_input=["ExtraOutput_4","ExtraOutput_5","ExtraOutput_6","ExtraOutput_7","ExtraOutput_8","ExtraOutput_9","ExtraOutput_10","ExtraOutput_11","ExtraOutput_12"], desired_target=["ExtraOutput_1","ExtraOutput_2","ExtraOutput_3"]): # add target function input that acts on target mapping to create value
    # TODO: add real commenting everywhere instead of pound signs
    '''
    Takes turbulence data filename, desired input parameters, target
    parameters, and a target construction function (if one wants to
    create a target value with target parameters), and stores data as
    a supervised dataset.

    # conservative 01: density
    #              02: density * x velocity
    #              03: density * y velocity
    #              04: density * z velocity
    #              05: nu tilda

    # extra output 01: turbulent production
    #              02: turbulent destruction
    #              03: turbulent cross production
    #              04: nu (normal kinematic viscosity)
    #              05: nu tilda
    #              06: wall distance
    #              07: d nu tilda dx
    #              08: d nu tilda dy
    #              09: du/dx
    #              10: du/dy
    #              11: dv/dx
    #              12: dv/dy
    #              13: 1 - 2 + 3 (constructed) as function of 4 <-> 12

    # dimensions-> (9) input, (1) output
    '''
    reynolds_base = re.findall(r'[0-9]+', target_file_name)
    reynolds_base = int(reynolds_base[0])
    self.reynolds_number = reynolds_base * 100000

    f = open(target_file_name, 'r')
    key_line  = f.readline()
    key_names = key_line.strip().split('\t')
    key_names = [ key_name[1:-1] for key_name in key_names ]

    input_mapping  = [ key_names.index(term) for term in desired_input ]
    target_mapping = [ key_names.index(term) for term in desired_target ]

    input_data  = []
    target_data = []
    for line in f.readlines():
      values = line.strip().split('\t')
      values = [ float(value) for value in values ]
      inp    = [ values[ind] for ind in input_mapping ]
      tgt    = [ values[ind] for ind in target_mapping ]
      tgt    = [ tgt[0] - tgt[1] + tgt[2] ]
      input_data.append(inp)
      target_data.append(tgt) # TODO: apply a target formation function instead
    f.close()

    self.in_dim   = len(input_data[0])
    self.out_dim  = len(target_data[0])
    self.tot_size = len(target_data)

    all_data = SupervisedDataSet(self.in_dim, self.out_dim)
    for ind in xrange(self.tot_size):
      in_datum  = input_data[ind]
      out_datum = target_data[ind]
      all_data.addSample(in_datum,out_datum)

    tst_data, trn_data = all_data.splitWithProportion(self.split_proportion)

    self.all_data = all_data
    self.tst_data = tst_data
    self.trn_data = trn_data
    self.portion  = {"training":None,"test":None}
























