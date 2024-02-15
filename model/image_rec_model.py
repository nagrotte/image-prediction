##iMAGE REC MODEL USING Transfer Learning Inception V3
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config= ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input,Lambda, Dense, Flatten
from tensorflow.keras.models import Model

