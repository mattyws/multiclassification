import csv
import json
import os
import shutil
import functions
from time import time

parameters = functions.load_parameters_file()


mimic_data_path = parameters['mimic_data_path']
csv_file_path = mimic_data_path + parameters['csv_files_directory']
files = [
    'OUTPUTEVENTS', 'CHARTEVENTS', 'MICROBIOLOGYEVENTS',
    'LABEVENTS', 'NOTEEVENTS', 'PRESCRIPTIONS', 'CPTEVENTS',
    'INPUTEVENTS_CV', 'INPUTEVENTS_MV'
]

def writeBuffer(writer, buffer):
    for data in buffer:
        writer.writerow(data)

def closeFilePointer(files_dict):
    for key in files_dict.keys():
        files_dict[key]['file'].close()

for file in files:
    print("========================================= {} ====================================================".format(file))
    data_path = mimic_data_path+file
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        start = time()
        print("Deleting created directory -----------------")
        shutil.rmtree(data_path)
        end = time()
        print("Took {} seconds to delete the directory.".format(end-start))
        os.mkdir(data_path)
    with open(csv_file_path+file+'.csv', 'r') as csv_source_file:
        reader = csv.DictReader(csv_source_file)
        buffer = []
        print("Iteration on file -----------------")
        start = time()
        total_rows = 0
        empty_admission = 0
        files_dict = dict()
        for row in reader:
            total_rows += 1
            if total_rows % 100000 == 0:
                print("{} total rows processed".format(total_rows))
            if len(row['HADM_ID']) == 0:
                empty_admission += 1
            #Write buffer
            file_path = data_path+"/{}_{}.csv".format(file, row["HADM_ID"])
            if row["HADM_ID"] not in files_dict.keys():
                files_dict[row["HADM_ID"]] = dict()
                files_dict[row["HADM_ID"]]['csv'] = file_path
                with open(file_path, 'w') as file_handler:
                    csv_writer = csv.DictWriter(file_handler, row.keys())
                    csv_writer.writeheader()
            with open(files_dict[row["HADM_ID"]]['csv'], 'a') as file_handler:
                csv_writer = csv.DictWriter(file_handler, row.keys())
                csv_writer.writerow(row)
        end = time()
        print("Took {} seconds to process file. Total rows processed {}, and {} rows with empty admission id.".
              format(end-start, total_rows, empty_admission))
        closeFilePointer(files_dict)