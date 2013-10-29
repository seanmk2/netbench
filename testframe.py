from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer
from pybrain.structure import SigmoidLayer
from pybrain.structure import TanhLayer
from pybrain.structure import FullConnection

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

  def net_activate(self, data):
    print self.neural_net.activate(data)

  def net_params(self):
    print self.neural_net.params
