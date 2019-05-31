"""
This script filters .json from patients that were indentified with sepsis by the Sepsis 3 criteria.
How to get the sepsis3 file can be found in the repository https://github.com/alistairewj/sepsis3-mimic
"""
import json
import os

import pandas as pd
from shutil import copyfile
import numpy as np

mimic_data_path = "/home/mattyws/Documentos/mimic/data/"
json_files_path = mimic_data_path+"json/"
sepsis3_json_files_path = mimic_data_path+"json_sepsis/"
nosepsis3_json_files_path = mimic_data_path+"json_sepsis/no_sepsis/"

sepsis3_df = pd.read_csv('sepsis3-df.csv')
sepsis3_df = sepsis3_df[sepsis3_df["sepsis-3"] == 1]

hadm_ids = sepsis3_df['hadm_id'].values

sepsis3_files = []


def add_nonsepsis_json():
    return np.random.randint(1, 10) % 2 == 0


added_non_sepsis = 0
files_visited = 0
for dir, path, files in os.walk(json_files_path):
    for file in files:
        files_visited += 1
        if files_visited % 1000 == 0:
            print("Visited files {}".format(files_visited))
        hadm_id_file = int(file.split('.')[0])
        if hadm_id_file in hadm_ids:
            hadm_sepsis3_rows = sepsis3_df[sepsis3_df['hadm_id'] == hadm_id_file]
            # print(hadm_sepsis3_rows.to_dict())
            json_admission = json.load(open(dir+'/'+file, 'r'))
            json_admission['sepsis3'] = hadm_sepsis3_rows.iloc[0].to_dict()
            for key in json_admission['sepsis3'].keys():
                if type(json_admission['sepsis3'][key]) != int and type(json_admission['sepsis3'][key]) != float:
                    json_admission['sepsis3'][key] = str(json_admission['sepsis3'][key])
            with open(sepsis3_json_files_path + file, 'w') as new_json_file_handler:
                json.dump(json_admission, new_json_file_handler, sort_keys=True, indent=4, separators=(', ', ': '))
        else:
            if added_non_sepsis < len(hadm_ids) * 1.5 and add_nonsepsis_json():
                copyfile(dir + "/" + file, nosepsis3_json_files_path + file)
                added_non_sepsis += 1