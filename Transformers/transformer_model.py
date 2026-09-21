import pandas as pd
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset

computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie' or computer == 'bethe':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')

import pandas as pd
import random
import numpy as np
from scipy.stats import zscore
import tqdm
import time

print(f'\nGPU availability: {torch.cuda.is_available()}')


data_directory = input('Data directory: ')
data_processes = int(input('Num. data processors: '))
max_epochs = int(input('Max epochs: '))
all_parquets = os.listdir(data_directory)
patience = int(input('Patience: '))
training_fraction = float(input('Enter training data fraction: '))
lower_energy_bound = float(input('Enter lower energy bound in eV: '))

n_training_samples = int(training_fraction * len(all_parquets))

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_parquets)
	if choice not in training_files:
		training_files.append(choice)

val_files = []
for file in all_parquets:
	if file not in training_files:
		val_files.append(file)

print('\nFetching training data...')



def fetch_data(datafile):
	"""Retrieves data from the parquet file and returns it as a matrix"""
	temp_df = pd.read_parquet(f'{data_directory}/{datafile}', engine='pyarrow')
	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]

	keff_value = float(temp_df['keff'].values[0])

	pu9_mt18xs = temp_df['94239_MT18_XS'].values.tolist()
	pu0_mt18xs = temp_df['94240_MT18_XS'].values.tolist()
	pu1_mt18xs = temp_df['94241_MT18_XS'].values.tolist()

	pu9_mt2xs = temp_df['94239_MT2_XS'].values.tolist()
	pu0_mt2xs = temp_df['94240_MT2_XS'].values.tolist()
	pu1_mt2xs = temp_df['94241_MT2_XS'].values.tolist()

	pu9_mt4xs = temp_df['94239_MT4_XS'].values.tolist()
	pu0_mt4xs = temp_df['94240_MT4_XS'].values.tolist()
	pu1_mt4xs = temp_df['94241_MT4_XS'].values.tolist()

	pu9_mt16xs = temp_df['94239_MT16_XS'].values.tolist()
	pu0_mt16xs = temp_df['94240_MT16_XS'].values.tolist()
	pu1_mt16xs = temp_df['94241_MT16_XS'].values.tolist()

	pu9_mt102xs = temp_df['94239_MT102_XS'].values.tolist()
	pu0_mt102xs = temp_df['94240_MT102_XS'].values.tolist()
	pu1_mt102xs = temp_df['94241_MT102_XS'].values.tolist()

	XS_obj = [pu9_mt2xs, pu9_mt4xs, pu9_mt16xs, pu9_mt18xs, pu9_mt102xs,
				pu0_mt2xs, pu0_mt4xs, pu0_mt16xs, pu0_mt18xs, pu0_mt102xs,
				pu1_mt2xs, pu1_mt4xs, pu1_mt16xs, pu1_mt18xs, pu1_mt102xs,]

	return(XS_obj, keff_value)



keff_train = [] # k_eff labels
XS_train = []

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, train_file) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, keff_value = future.result()
		XS_train.append(xs_values)
		keff_train.append(keff_value)

XS_train = np.array(XS_train) # shape (num_samples, num_channels, points per channel)

keff_train_mean = np.mean(keff_train)
keff_train_std = np.std(keff_train)
y_train = zscore(keff_train)

XS_val = []
keff_val = []

print('\nFetching validation data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, val_file) for val_file in val_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, keff_value_val = future.result()
		XS_val.append(xs_values_val)
		keff_val.append(keff_value_val)

XS_val = np.array(XS_val)
y_val = (np.array(keff_val) - keff_train_mean) / keff_train_std


print('\nScaling training and validation data...')


# le_bound_index = 1 # filters out NaNs

def process_data(XS_train, XS_val):


	channel_matrix_train = [[] for i in range(len(XS_train[0]))] # each element is a matrix of only one channel, e.g. channel_matrix[0] is all the lists containing
	channel_matrix_val = [[] for i in range(len(XS_val[0]))]

	scaled_channel_matrix_train = []
	scaled_channel_matrix_val = []

	for matrix in tqdm.tqdm(XS_train, total =len(XS_train)):
		# matrix has shape (num channels, points per channel)
		for channel_index, channel in enumerate(matrix):
			channel_matrix_train[channel_index].append(channel)

		# channel_matrix now has shape (num channels, num samples, points per channel)
		# Each element of channel matrix has shape (num samples, points per channel)
	for matrix in tqdm.tqdm(XS_val, total =len(XS_val)):
		for channel_index, channel in enumerate(matrix):
			channel_matrix_val[channel_index].append(channel)

	#################################################################################################################################################################
	for channel_data_train, channel_data_val in zip(channel_matrix_train, channel_matrix_val): # each iterative variable is the tensor of one specific channel e.g. Pu-239 fission, for all samples
		transposed_matrix_train = np.transpose(channel_data_train) # shape (energy points per sample, num samples)
		transposed_matrix_val = np.transpose(channel_data_val)

		transposed_scaled_channel_train = []
		transposed_scaled_channel_val = []
		for energy_point_train, energy_point_val in zip(transposed_matrix_train[:-1], transposed_matrix_val[:-1]): # each point on the unionised energy grid

			train_mean = np.mean(energy_point_train)
			train_std = np.std(energy_point_train)

			scaled_point_train = zscore(energy_point_train)
			transposed_scaled_channel_train.append(scaled_point_train)

			scaled_point_val = (np.array(energy_point_val) - train_mean) / train_std
			transposed_scaled_channel_val.append(scaled_point_val)

		scaled_channel_train = np.array(transposed_scaled_channel_train)
		scaled_channel_train = scaled_channel_train.transpose()
		scaled_channel_matrix_train.append(scaled_channel_train)

		scaled_channel_val = np.array(transposed_scaled_channel_val)
		scaled_channel_val = scaled_channel_val.transpose()
		scaled_channel_matrix_val.append(scaled_channel_val)

	###################################################################################################################################################################
	# print('Forming scaled training data...')
	X_matrix_train = [[] for i in range(XS_train.shape[0])] # number of samples
	X_matrix_val = [[] for i in range(XS_val.shape[0])]

	for scaled_observable in scaled_channel_matrix_train:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_train[sample_index].append(channel_sample)

	for scaled_observable in scaled_channel_matrix_val:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_val[sample_index].append(channel_sample)

	X_matrix_train = np.array(X_matrix_train)
	X_matrix_train[np.isnan(X_matrix_train)] = 0

	X_matrix_val = np.array(X_matrix_val)
	X_matrix_val[np.isnan(X_matrix_val)] = 0 # changes nans to 0

	return X_matrix_train, X_matrix_val

X_train, X_val = process_data(XS_train, XS_val)


# From here switch from numpy matrices to torch tensors
print('\nConverting arrays to tensors...')

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()

# Working with tensors now
print('\nDefining torch classes and functions...')

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super().__init__()

		positional_encoding = torch.zeros(max_len, d_model) # matrix of zeroes with dimensions (max_len, d_model)

		position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # indices for positions

		divisor_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

		positional_encoding[:, 0::2] = torch.sin(position * divisor_term) # apply sin encoding to even positions

		positional_encoding[:, 1::2] = torch.cos(position * divisor_term) # apply cos encoding to odd positions

		positional_encoding = positional_encoding.unsqueeze(0)

		self.register_buffer('positional_encoding', positional_encoding)


	def forward(self, x):
		"""x shape (batch, seq_len, d_model)"""
		# get current sequence length of  input
		sequence_length = x.size(1)

		# applies positional encoding to the input embeddings
		return x + self.positional_encoding[:, :sequence_length, :]

class RegressionTransformerFeatureRows(nn.Module):
	def __init__(self,
				 num_features: int,
				 d_model: int = 64,
				 nhead: int = 4,
				 num_layers: int = 2,
				 dim_feedforward: int = 128,
				 dropout: float = 0.1,
				 max_len: int = 5000):
		super().__init__()

		self.num_features = num_features

		# model input of shape (batch, num_features, sequence_length)
		self.input_proj = nn.Linear(num_features, d_model)

		# apply positional encoding to tokens
		self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

		# single encoder layer with
		# multi-head self-attention, feedforward network, layer norm, residual connections

		# transformer encoder layer
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True)  # ensures input tensors are (batch, seq, d_model)

		# stack multiple encoder layers into full transformer encoder
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer,	num_layers=num_layers)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer,	num_layers=num_layers)

		# regression head produces a single scalar (k_eff) output
		# after encoding, sequence is pooled into a single vector of size d_model
		# sequential vanilla mlp outputs scalar value in scaled k_eff space
		self.regression_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

	def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
		"""x: (batch, num_features, seq_len)
		src_key_padding_mask: (batch, seq_len) with True where padding exists (optional)"""

		model_device = self.input_proj.weight.device
		# print('Model device:', model_device)
		x = x.to(model_device)

		# re-orient input to make columns/tokens
		# rows = features and columns = tokens, so transpose
		# before: (batch, num_features, seq_len) after:  (batch, seq_len, num_features)

		x = x.transpose(1, 2)
		# linear projection: embed tokens into d_model-dimensional space
		# every token now becomes a d_model-dimensional embedding
		x = self.input_proj(x)  # (batch, seq_len, d_model)

		# add positional encoding
		x = self.pos_encoder(x)

		# transformer encoding including
		x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # ignore padded tokens
		# (batch, seq_len, d_model)

		if src_key_padding_mask is not None: # pool sequence into single vector for regression head
			# mask aware mean pooling (if padding used) or simple mean pooling (no padding)
			# mask: True = padded token, False = real token
			mask = ~src_key_padding_mask  # invert: True = valid token
			mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)

			# zero out padded token embeddings
			x = x * mask

			# sum over valid tokens only
			summed = x.sum(dim=1)  # (batch, d_model)
			counts = mask.sum(dim=1).clamp(min=1)
			pooled = summed / counts
		else:
			# average over all tokens
			pooled = x.mean(dim=1)  # (batch, d_model)

		# mlp regression head
		out = self.regression_head(pooled)  # (batch, 1)

		# return shape (batch) instead of (batch, 1)
		return out.squeeze(-1)


class EarlyStopping:
	"""stops training early if requirements not met"""
	def __init__(self, patience, min_delta=0.0, mode="min", restore_best_weights=True):
		assert mode in ("min", "max")
		self.patience = patience # minimum epochs to wait for improvement
		self.min_delta = min_delta  # minimum change in loss (minimising loss)
		self.mode = mode # set whether metric is being minimised or maximised
		self.restore_best_weights = restore_best_weights # restore model params for best epoch

		self.best_score = None
		self.best_state = None
		self.bad_epochs = 0
		self.should_stop = False

	def _improved(self, score, best): # sets mode
		if self.mode == "min":
			return score < (best - self.min_delta)
		else:
			return score > (best + self.min_delta)

	def step(self, score, model):
		if self.best_score is None: # save model params for best epoch
			self.best_score = score
			self.best_state = copy.deepcopy(model.state_dict())
			return

		if self._improved(score, self.best_score):
			self.best_score = score
			self.best_state = copy.deepcopy(model.state_dict())
			self.bad_epochs = 0
		else:
			self.bad_epochs += 1
			if self.bad_epochs >= self.patience:
				self.should_stop = True

	def restore(self, model):
		if self.restore_best_weights and self.best_state is not None:
			model.load_state_dict(self.best_state)

def iter_minibatches(X, y, batch_size, shuffle=False, device=None):
	"""yields (X_batch, y_batch) with X_batch shape (B, F, T)
	y_batch shape (B)"""
	n = X.shape[0]
	idx = torch.randperm(n) if shuffle else torch.arange(n)

	for start in range(0, n, batch_size):
		batch_idx = idx[start:start + batch_size]
		Xb = X[batch_idx]
		yb = y[batch_idx]

		if device is not None:
			Xb = Xb.to(device, non_blocking=True)
			yb = yb.to(device, non_blocking=True)

		yield Xb, yb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
N_train, F, T = X_train.shape # F stands for features i.e. number of reaction channels (T stands for tokens, each token is an energy point)
N_val = X_val.shape[0]

# define model
model = RegressionTransformerFeatureRows(num_features=F, max_len=T).to(device)

# standard loss/optimisation
criterion = nn.MSELoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

print('\nDefining early stopping criteria...')

early = EarlyStopping(patience=patience, min_delta=1e-5, mode="min", restore_best_weights=True)

batch_size = 32 # standard


print('\nBeginning training...')

for epoch in tqdm.tqdm(range(1, max_epochs + 1)):
	# start training
	model.train()
	train_loss_sum = 0.0
	train_count = 0

	for Xb, yb in iter_minibatches(X_train, y_train, batch_size, shuffle=True, device=device):
		optimiser.zero_grad(set_to_none=True)
		# print("Xb device:", Xb.device)
		# print("model device:", next(model.parameters()).device)

		preds = model(Xb)              # (B,)
		loss = criterion(preds, yb)    # scalar

		loss.backward()
		optimiser.step()

		train_loss_sum += loss.item() * Xb.size(0)
		train_count += Xb.size(0)

	train_loss = train_loss_sum / max(train_count, 1)

	# validation
	model.eval()
	val_loss_sum = 0.0
	val_count = 0

	with torch.no_grad():
		for Xb, yb in iter_minibatches(X_val, y_val, batch_size=64, shuffle=False, device=device):
			preds = model(Xb)
			loss = criterion(preds, yb)

			val_loss_sum += loss.item() * Xb.size(0)
			val_count += Xb.size(0)

	val_loss = val_loss_sum / max(val_count, 1)

	print(f"Epoch {epoch:03d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")

	# early stopping
	early.step(val_loss, model)
	if early.should_stop:
		print(f"Early stopping. Best val MSE: {early.best_score:.6f}")
		break

# restore best weights after training
early.restore(model)





def timed_eval(model, inputs):
	"""time evals"""
	start = time.perf_counter()
	with torch.no_grad():
		out = model(inputs)
	end = time.perf_counter()
	return out, end - start


model.eval()  # switch to inference mode

with torch.no_grad():  # improve compute cost disable gradient tracking
	predictions = model(X_val.to(device))  # forward pass

out, timing = timed_eval(model, X_val)

rescaled_predictions = []
predictions_list = predictions.tolist()

for pred in predictions_list:
	descaled_p = pred * keff_train_std + keff_train_mean
	rescaled_predictions.append(float(descaled_p))

errors = []
for predicted, true in zip(rescaled_predictions, keff_val):
	errors.append((predicted - true) * 1e5)
	print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

sorted_errors = sorted(errors)
absolute_errors = [abs(x) for x in sorted_errors]
print(f'Average absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')

print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')

print(f"Smallest absolute error: {min(absolute_errors)} pcm")
acceptable_predictions = []
borderline_predictions = []
twenty_pcm_predictions = []
for x in absolute_errors:
	if x <= 5.0:
		acceptable_predictions.append(x)
	if x <= 10.0:
		borderline_predictions.append(x)
	if x <= 20.0:
		twenty_pcm_predictions.append(x)

print(f' {len(acceptable_predictions)} ({len(acceptable_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 5 pcm error')
print(f' {len(borderline_predictions)} ({len(borderline_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 10 pcm error')
print(f' {len(twenty_pcm_predictions)} ({len(twenty_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 20 pcm error)')
