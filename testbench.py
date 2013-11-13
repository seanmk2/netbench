import testframe

in_size     = 10
hidden_size = 50
out_size    = 3
num_samples = 5000

my_net  = testframe.NeuralNet(in_size, hidden_size, out_size)
my_data = testframe.RandomDataSet(in_size, out_size, num_samples)
my_net.train_many_k_means_reductions(my_data, iters=20, k_means_reductions=[0.25,0.50,0.75])

### add length of time to reduce data, vs. time to train reduced net
