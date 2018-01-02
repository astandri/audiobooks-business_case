from dataset_preparation import DatasetPreparation
from data_reader import DataReader
from model import Model

def main():
	#Prepare dataset from csv to npz files
	DatasetPreparation.prepare('Audiobooks-data.csv')
	
	#Read the dataset, create batches, and one hot encode the targets
	batch_size = 100
	train_data = DataReader('Audiobooks_data_train.npz',batch_size)
	validation_data = DataReader('Audiobooks_data_validation.npz')
	test_data = DataReader('Audiobooks_data_test.npz')	
	
	m = Model(train_data,validation_data)
	m.train()
	
	
	
  
if __name__== "__main__":
	main()
