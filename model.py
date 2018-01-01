import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf 

from dataset_preparation import DatasetPreparation

def main():
	DatasetPreparation.prepare('Audiobooks-data.csv')
  
if __name__== "__main__":
	main()
