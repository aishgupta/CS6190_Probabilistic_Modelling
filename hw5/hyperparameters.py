import torch

max_epoch       = 1000
workers         = 4
train_batch_size= 872
test_batch_size = 500
log_iter        = 100

TOLERANCE    = 1e-5
input_dim    = 5
n_class      = 2
SAMPLES      = 100
TEST_SAMPLES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = torch.device("cpu")

lr         = 0.001
n_hidden   = 20
activation = 'relu'
