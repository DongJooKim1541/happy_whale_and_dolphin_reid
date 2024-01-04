from torch.nn import PairwiseDistance

batch_size=64
num_train_triplets=40000
margin = 0.0001
epochs = 100
learning_rate = 1e-4

l2_distance = PairwiseDistance(p=2)