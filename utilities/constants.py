import torch

# Taken from the paper
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

SCHEDULER_WARMUP_STEPS  = 4000

DROPOUT_P               = 0.1

SEPERATOR               = "========================="

if(torch.cuda.device_count() > 0):
    TORCH_DEVICE = torch.device("cuda")
else:
    print("WARNING: Using the cpu. This will cause the model to run very slow!")
    TORCH_DEVICE = torch.device("cpu")

TORCH_FLOAT             = torch.float32
