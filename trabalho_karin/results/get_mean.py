import csv
import os
import pandas as pd

files = os.listdir('./')
mean_file = 'mean_results.csv'
with open(mean_file, 'w') as mean_file_handler:
    dict_writer = None
    means = []
    for file in files:
        if '.csv' in file and file != mean_file:
            with open(file, 'r') as file_handler:
                data = pd.read_csv(file_handler)
                data_means = data.mean() * 100
                data_means = data_means.to_dict()
                data_means["file"] = file.split('.')[0]
                if dict_writer is None:
                    dict_writer = csv.DictWriter(mean_file_handler, data_means.keys())
                    dict_writer.writeheader()
                dict_writer.writerow(data_means)
