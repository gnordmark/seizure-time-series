
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv('SeizureData.csv')

#print("hello world!")
#print(data.head())

data.drop('Unnamed', axis=1, inplace=True) #First column is worthless metadata

# Use binary classification: seizure(y=1) vs no seizure (y=2-5)
binary_labels = {1:1, 2:0, 3:0, 4:0, 5:0}
data.replace({'y': binary_labels}, inplace=True)

# Int values of seizures and non-seizures 
seizure_counts = data['y'].value_counts()
num_seizures = seizure_counts[1]
num_non_seizure = seizure_counts[0]

print(f"Number of seizures {num_seizures} vs non-seizures {num_non_seizure}")

# DataFrames of seizures and non-seizures
non_seizures = data[data['y'] == 0]
seizures = data[data['y'] == 1]


# data is in 1s per row format
# first unpivot into single time series, preserve target y, then take the original index, which is the
# "sample index" of each sample
data_unpivoted = (data
                  .melt(id_vars=['y'], var_name='time_label', value_name='eeg', ignore_index=False)
                  .reset_index()
                  .rename(columns={'index': 'sample_index'})
                  )

# the time index is the index over the 1s time period in each original row in data
data_unpivoted['time_index'] = (data_unpivoted['time_label']
                                .str.extract(r'(\d+)', expand=False)
                                .astype(int)
                                )

# sort each window according to the sample and time and re-order columns
data_unpivoted = (data_unpivoted
                  .sort_values(by=['sample_index', 'time_index'])
                  .reindex(['sample_index', 'time_index', 'eeg', 'y'], axis=1)
                  )

print(data_unpivoted)