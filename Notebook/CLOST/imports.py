import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
#import forecastio
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Conv3D, Flatten, Dropout, MaxPooling3D, Activation, Lambda, Reshape, Concatenate, LSTM
import keras.backend as K
from functools import reduce
import h5py
from keras.backend import sigmoid
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

import warnings
warnings.filterwarnings("ignore")