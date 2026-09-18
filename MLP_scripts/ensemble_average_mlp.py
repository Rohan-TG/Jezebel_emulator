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
data_directory = input('\n\nTraining data directory: ')
test_directory = input('\nTest data directory: ')
test_files = os.listdir(test_directory)


all_parquets = os.listdir(data_directory)

training_fraction = float(input('\nEnter training data fraction: '))
lower_energy_bound = float(input('\nEnter lower energy bound in eV: '))
patience = float(input('\nEnter patience: '))
n_models = int(input('\nN. models: '))
keep_n = int(input('\nKeep n. models: '))


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

if test_directory == 'x':
	test_files = val_files
print('\nFetching training data...')

start_time = time.time()


average_performance_list = []
average_performance_list_test = []
pcm10_performance_list = []

def fetch_data(datafile, dir = data_directory):

	temp_df = pd.read_parquet(f'{dir}/{datafile}', engine='pyarrow')
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

	xsobject = pu9_mt2xs + pu9_mt4xs + pu9_mt16xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt16xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt4xs + pu1_mt16xs + pu1_mt18xs + pu1_mt102xs

	XS_obj = xsobject

	return(XS_obj, keff_value)



keff_train = [] # unscaled k_eff train labels
XS_train = [] # unscaled xs data

pu9_train_indices = []
pu0_train_indices = []
pu1_train_indices = []

for train_file in tqdm.tqdm(training_files, total=n_training_samples):
	xs_values, keff_value = fetch_data(train_file)

	XS_train.append(xs_values)
	keff_train.append(keff_value)

	pu9_train_index = int(train_file.split('_')[4])
	pu9_train_indices.append(pu9_train_index)

	pu0_train_index = int(train_file.split('_')[6])
	pu0_train_indices.append(pu0_train_index)

	pu1_train_index = int(train_file.split('_')[8].split('.')[0])
	pu1_train_indices.append(pu1_train_index)

XS_train = np.array(XS_train)
y_train = zscore(keff_train)

train_labels_mean = np.mean(keff_train)
train_labels_std = np.std(keff_train)


scaling_matrix_xtrain = XS_train.transpose()

scaled_columns_xtrain = []
print('\nScaling training data...')


le_bound_index = 1 # filters out NaNs


training_column_means = []
training_column_stds = []

for column in tqdm.tqdm(scaling_matrix_xtrain[le_bound_index:-1], total=len(scaling_matrix_xtrain[le_bound_index:-1])):
	scaled_column = zscore(column)

	column_mean = np.mean(column)
	column_std = np.std(column)
	training_column_means.append(column_mean)
	training_column_stds.append(column_std)
	scaled_columns_xtrain.append(scaled_column)

scaled_columns_xtrain = np.array(scaled_columns_xtrain)
X_train = scaled_columns_xtrain.transpose()


XS_val = []
keff_val = []

print('\nFetching val data...')

pu9_val_indices = []
pu0_val_indices = []
pu1_val_indices = []

for val_file in tqdm.tqdm(val_files, total=len(val_files)):
	xs_values_val, keff_value_val = fetch_data(val_file)
	XS_val.append(xs_values_val)
	keff_val.append(keff_value_val)

	pu9_val_index = int(val_file.split('_')[4])
	pu9_val_indices.append(pu9_val_index)

	pu0_val_index = int(val_file.split('_')[6])
	pu0_val_indices.append(pu0_val_index)

	pu1_val_index = int(val_file.split('_')[8].split('.')[0])
	pu1_val_indices.append(pu1_val_index)

XS_val = np.array(XS_val)
y_val = (np.array(keff_val) - train_labels_mean) / train_labels_std

scaling_matrix_xval = XS_val.transpose()

scaled_columns_xval = []
print('\nScaling val data...')
for column, c_mean, c_std in tqdm.tqdm(zip(scaling_matrix_xval[le_bound_index:-1], training_column_means, training_column_stds), total=len(scaling_matrix_xval[le_bound_index:-1])):

	scaled_column = (np.array(column) - c_mean) / c_std
	scaled_columns_xval.append(scaled_column)

scaled_columns_xval = np.array(scaled_columns_xval)
X_val = scaled_columns_xval.transpose()



print('\nFetching test data...')

def generate_test_data(test_data_directory, mask=None, masking_value=0, truncate=False):
	files = os.listdir(test_data_directory)

	limiter = 795

	keff_test = []
	XS_test = []
	pu9_test_indices = []
	pu0_test_indices = []
	pu1_test_indices = []
	for test_file in tqdm.tqdm(files, total=len(files)):
		xs_values, keff_value = fetch_data(test_file, dir=test_data_directory)

		if truncate:
			ticker = 0
			for val_index, val in enumerate(xs_values):
				if ticker < 1248:
					ticker += 1
				elif ticker >= 1248:
					ticker = 0
				if ticker < limiter:
					xs_values[val_index] = 0


		XS_test.append(xs_values)
		keff_test.append(keff_value)

		pu9_test_index = int(test_file.split('_')[4])
		pu9_test_indices.append(pu9_test_index)

		pu0_test_index = int(test_file.split('_')[6])
		pu0_test_indices.append(pu0_test_index)

		pu1_test_index = int(test_file.split('_')[8].split('.')[0])
		pu1_test_indices.append(pu1_test_index)

	XS_test = np.array(XS_test)
	y_test = (np.array(keff_test) - train_labels_mean) / train_labels_std

	scaling_matrix_xtest = XS_test.transpose()

	scaled_columns_xtest = []
	print('\nScaling test data...')
	for column, c_mean, c_std in tqdm.tqdm(
			zip(scaling_matrix_xtest[le_bound_index:-1], training_column_means, training_column_stds),
			total=len(scaling_matrix_xtest[le_bound_index:-1])):
		scaled_column = (np.array(column) - c_mean) / c_std
		scaled_columns_xtest.append(scaled_column)

	scaled_columns_xtest = np.array(scaled_columns_xtest)
	X_test = scaled_columns_xtest.transpose()
	X_test = np.nan_to_num(X_test, nan=0.0)

	if mask is not None:
		X_test[np.abs(X_test) >= float(mask)] = masking_value

	return(X_test, y_test, keff_test)

X_test, y_test, keff_test = generate_test_data(test_data_directory=test_directory)
X_test = np.nan_to_num(X_test, nan=0.0)

# remove NaNs
X_val = np.nan_to_num(X_val, nan=0.0)
X_train = np.nan_to_num(X_train, nan=0.0)






def build_model():
	"""Returns a Keras Sequential model"""
	callback = keras.callbacks.EarlyStopping(monitor='val_loss',
											 patience=patience,
											 mode='min',
											 restore_best_weights=True)

	model =keras.Sequential()
	model.add(keras.layers.Dense(1000, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
	model.add(keras.layers.Dense(900, activation='relu'))
	model.add(keras.layers.Dense(750, activation='relu'))
	model.add(keras.layers.Dense(600, activation='relu'))
	model.add(keras.layers.Dense(540, activation='relu'))
	model.add(keras.layers.Dense(480, activation='relu'))
	model.add(keras.layers.Dense(380, activation='relu'))
	model.add(keras.layers.Dense(250, activation='relu'))
	model.add(keras.layers.Dense(200, activation='relu'))
	model.add(keras.layers.Dense(1, activation='linear'))
	model.compile(loss='MeanSquaredError', optimizer='adam')

	return model, callback

model_list = []

prediction_matrix = [[] for i in range(len(y_val))]
error_matrix_val = [[] for i in range(len(y_val))]

prediction_matrix_test = [[] for i in range(len(y_test))]
error_matrix_test = [[] for i in range(len(y_test))]

prediction_matrix_train = [[] for i in range(len(y_train))]
error_matrix_train = [[] for i in range(len(y_train))]

save_models = []

for num in tqdm.tqdm(range(n_models)):
	keras.backend.clear_session()

	temp_model, callback = build_model()



	trainstart = time.time()
	history = temp_model.fit(X_train,
						y_train,
						epochs=10000,
						batch_size=32,
						callbacks=callback,
						validation_data=(X_val, y_val),
						verbose=1)

	train_end = time.time()
	print(f'\nTraining completed in {datetime.timedelta(seconds=(train_end - trainstart))}')

	##################################### Predictions on test data #####################################################
	predictions_test = temp_model.predict(X_test)
	predictions_test = predictions_test.ravel()
	rescaled_predictions_test = []
	predictions_list_test = predictions_test.tolist()
	for p in predictions_list_test:
		descaled_p = p * train_labels_std + train_labels_mean
		rescaled_predictions_test.append(float(descaled_p))


	errors_test = []
	for predicted, true in zip(rescaled_predictions_test, keff_test):
		errors_test.append((predicted - true) * 1e5)

	for p_i, p in enumerate(rescaled_predictions_test):
		prediction_matrix_test[p_i].append(p)

	for err_index, err in enumerate(errors_test):
		error_matrix_test[err_index].append(err)


	absolute_errors_test = [abs(x) for x in errors_test]
	average_performance_list_test.append(np.mean(absolute_errors_test))

	####################################################################################################################
	predictions_train = temp_model.predict(X_train)
	predictions_train = predictions_train.ravel()
	rescaled_predictions_train = []
	predictions_list_train = predictions_train.tolist()

	for predtrain in predictions_list_train:
		descaled_p = predtrain * train_labels_std + train_labels_mean
		rescaled_predictions_train.append(float(descaled_p))

	errors_train = []
	for predictedtrain, truetrain in zip(rescaled_predictions_train, keff_train):
		errors_train.append((predictedtrain - truetrain) * 1e5)
		# print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

	# Save data into matrices
	for p_index, p in enumerate(rescaled_predictions_train):
		prediction_matrix_train[p_index].append(p)

	for err_index, err in enumerate(errors_train):
		error_matrix_train[err_index].append(err)

	####################################################################################################################
	predictions = temp_model.predict(X_val)
	predictions = predictions.ravel()


	rescaled_predictions = []
	predictions_list = predictions.tolist()

	for pred in predictions_list:
		descaled_p = pred * train_labels_std + train_labels_mean
		rescaled_predictions.append(float(descaled_p))

	errors = []
	for predicted, true in zip(rescaled_predictions, keff_val):
		errors.append((predicted - true) * 1e5)
		# print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

	# Save data into matrices
	for p_index, p in enumerate(rescaled_predictions):
		prediction_matrix[p_index].append(p)

	for err_index, err in enumerate(errors):
		error_matrix_val[err_index].append(err)

	sorted_errors = sorted(errors)
	absolute_errors = [abs(x) for x in sorted_errors]
	print(f'\nAverage absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')
	average_performance_list.append(np.mean(absolute_errors))

	print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')


	print(f"Smallest absolute error: {min(absolute_errors)} pcm")
	acceptable_predictions = []
	borderline_predictions = []
	fifteen_pcm_predictions = []
	twenty_pcm_predictions = []
	for x in absolute_errors:
		if x <= 5.0:
			acceptable_predictions.append(x)
		if x <= 10.0:
			borderline_predictions.append(x)
		if x <= 15.0 :
			fifteen_pcm_predictions.append(x)
		if x <= 20.0:
			twenty_pcm_predictions.append(x)
	pcm10_performance_list.append(len(borderline_predictions) / len(absolute_errors) * 100)

	print(f' {len(acceptable_predictions)} ({len(acceptable_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 5 pcm error')
	print(f' {len(borderline_predictions)} ({len(borderline_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 10 pcm error')
	print(f' {len(fifteen_pcm_predictions)} ({len(fifteen_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 15 pcm error)')
	print(f' {len(twenty_pcm_predictions)} ({len(twenty_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 20 pcm error)')


	# plt.figure()
	# plt.hist(sorted_errors, bins=30)
	# plt.grid()
	# plt.title(f'Session {num} Distribution of errors')
	# plt.xlabel('Error / pcm')
	# plt.ylabel('Count')
	# # plt.savefig('absolute_errors_corrected_scaling.png', dpi = 300)
	# plt.show()




	skew_positive = []
	skew_negative = []

	for x in errors:
		if x >0:
			skew_positive.append(x)
		else:
			skew_negative.append(x)

	# plt.figure()
	# plt.plot(keff_val, errors, 'x')
	# plt.grid()
	# plt.title(f'Session {num} Distribution of errors')
	# plt.xlabel('True k_eff')
	# plt.ylabel('Error / pcm')
	# # plt.savefig('errors_as_function_of_keff.png', dpi = 300)
	# plt.show()

	save_models.append(temp_model)

	# del history
	# del temp_model
	# gc.collect()

end_time = time.time()
import datetime
print(f'\nTotal runtime: {datetime.timedelta(seconds = (end_time - start_time))}')

def select_best_models(error_matrix, keep_n_models, threshold=10, mode='test'):
	"""error_matrix: the error matrix
	keep_n_models: the number of models to keep"""

	truncated_count_threshold = 0
	em = np.array(error_matrix)
	em_trans = np.transpose(em)

	model_averages = []
	for model_ in em_trans:
		model_averages.append(np.mean(np.abs(model_)))

	sorted_models = [[-1,100]]
	for i, val in enumerate(model_averages):
		for j, saved in enumerate(sorted_models):
			if val < saved[-1]:
				sorted_models.insert(j, [i,val])
				break

	acceptable_models = []
	for x in sorted_models[:keep_n_models]:
		acceptable_models.append(x[0])

	if mode == 'test':
		emt = np.array(error_matrix_test)
	else:
		emt = np.array(error_matrix_val)

	best_averaged_errors = []

	for sample in emt:
		working_list = []
		for model_index, value in enumerate(sample):
			if model_index in acceptable_models:
				working_list.append(value)

		if np.abs(np.mean(working_list)) <= threshold:
			truncated_count_threshold +=1

		best_averaged_errors.append(np.mean(working_list))


	return acceptable_models, truncated_count_threshold, best_averaged_errors

best_models, best_models_count10, averaged_errors = select_best_models(error_matrix_test, keep_n, mode='test')
print(best_models_count10 / len(keff_test) * 100)

selected_best_models = []
for k, model in enumerate(save_models):
	if k in best_models:
		selected_best_models.append(model)

