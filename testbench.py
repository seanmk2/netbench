import testframe

in_size     = 10
hidden_size = 50
out_size    = 3
num_samples = 333

my_net  = testframe.NeuralNet(in_size, hidden_size, out_size)
my_data = testframe.RandomDataSet(in_size, out_size, num_samples)
#my_net.train_many_k_means_reductions(my_data, iters=500, k_means_reductions=[0.01,0.03,0.10,0.30])
my_net.train_many_pca_reductions(my_data, iters = 333, pca_reductions=[1,2,3,4,10])

### add length of time to reduce data, vs. time to train reduced net
### also try naively taking data portions, compare to kmeans
### investigate using my own error function, aka RMSE
### neurolab might be a better option?

### train vs. test error on same graph, dash no dash
### investigate output function
### cut off for outliers outside of 3 std perhaps
### also graph data in pca 2 dimenions
### weighted k-means: bigger clusters contribute more sample points

# conservative 1: density
#              2: density * x velocity
#              3: density * y velocity
#              4: density * z velocity
#              5: nu tilda

# extra output 1:  turbulent production
#              2:  turbulent destruction
#              3:  turbulent cross production
#              4:  nu (normal kinematic viscosity)
#              5:  nu tilda
#              6:  wall distance
#              7:  d nu tilda dx
#              8:  d nu tilda dy
#              9:  du/dx
#              10: du/dy
#              11: dv/dx
#              12: dv/dy
#              13: 1 - 2 + 3 (constructed) as function of 4 <-> 12

#              9 input 1 output

#              replicate spalart almaras turbulent model
#              turbmodels.larc.nasa.gov/spalart.html


# parameters [][][]float64 layer/neuron/parameter
# parametersSlice []float64 parameter (look at NewPerParameterMemory())

# try quasi newton method, bfgs, lbfgs, conjugate gradient method (pyopt)