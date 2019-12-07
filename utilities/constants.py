import torch

from third_party.midi_processor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_VEL, RANGE_TIME_SHIFT

SEPERATOR               = "========================="

# Taken from the paper
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

SCHEDULER_WARMUP_STEPS  = 4000

DROPOUT_P               = 0.1

TOKEN_START             = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_END               = TOKEN_START + 1

VOCAB_SIZE              = TOKEN_END + 1

TORCH_CPU               = torch.device("cpu")
if(torch.cuda.device_count() > 0):
    TORCH_DEVICE = torch.device("cuda")
else:
    print(SEPERATOR)
    print("WARNING: Using the cpu. This will cause the model to run very slow!")
    print(SEPERATOR)
    print("")
    TORCH_DEVICE = TORCH_CPU

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long
