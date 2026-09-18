import os
import matplotlib.pyplot as plt
import keras.backend
import pickle
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore, norm
import tqdm
import keras
import time
import tensorflow as tf
import datetime

print(tf.config.list_physical_devices('GPU')) # ensure GPU is in use (optional)