import numpy as np
from sklearn import preprocessing

class DatasetPreparation:
	def __init__(self):
		self.type = 'process'

	def prepare():
		raw_csv = np.loadtxt('Audiobooks-data.csv',delimiter=',')

		unscaled_inputs_all = raw_csv[:,1:-1]
		targets_all = raw_csv[:,-1]
		
		#Balancing Dataset
		num_one_targets = int(np.sum(targets_all))
		zero_targets_counter = 0
		indices_to_remove = []

		for i in range(targets_all.shpe[0]):
			if targets_all[i] == 0
				zero_targets_counter += 1
				if zero_targets_counter > num_one_targets:
					indices_to_remove.append(i)
