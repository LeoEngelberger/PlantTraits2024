import os
import datetime
os.environ["KERAS_BACKEND"] = "torch"
import torch
from torch import cuda
from CustomModel import PlantGuesser
import LossFunctions


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Config import Config

print(torch.cuda.is_available())
print(cuda.cudaStatus)
print(torch.version.cuda)

model = PlantGuesser()
model.data_builder.check_dataset()
model.compile()
model.examine()