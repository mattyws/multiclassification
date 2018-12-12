import csv
import json
import os
import datetime
import pprint

from data_fields import DATA_FIELDS, D_ITEMS_RELATION, D_LABITEMS_RELATION

pp = pprint.PrettyPrinter(indent=5)
mimic_data_path = "/home/mattyws/Documentos/mimic/data/"
csv_file_path = mimic_data_path+"csv/"
json_files_path = mimic_data_path+"json/"
if not os.path.exists(json_files_path):
    os.mkdir(json_files_path)

# Static files for admission and dictionaries
admissions_csv_path = csv_file_path+"ADMISSIONS.csv"
d_items_csv_path = csv_file_path+'D_ITEMS.csv'
d_labitems_csv_path = csv_file_path+'D_LABITEMS.csv'
patient_csv_path = csv_file_path+'PATIENTS.csv'

paths = [
    'OUTPUTEVENTS', 'CHARTEVENTS', 'PROCEDURES_ICD', 'MICROBIOLOGYEVENTS',
    'LABEVENTS', 'DIAGNOSES_ICD', 'NOTEEVENTS', 'PRESCRIPTIONS', 'CPTEVENTS',
    'INPUTEVENTS_CV', 'INPUTEVENTS_MV'
]

with open(admissions_csv_path, 'r') as admissions_csv_file:
    # Open and reading dictionaries
    d_items = dict()
    d_labitems = dict()
    patients = dict()
    # CREATING D_ITEMS
    with open(d_items_csv_path, 'r') as d_items_file:
        d_items_dictreader = csv.DictReader(d_items_file)
        for row in d_items_dictreader:
            d_items[row['ITEMID']] = {}
            d_items[row['ITEMID']]['ITEMID'] = row['ITEMID']
            d_items[row['ITEMID']]['LABEL'] = row['LABEL']

    #CREATING D_LABITEMS
    with open(d_labitems_csv_path, 'r') as d_labitems_file:
        d_labitems_dictreader = csv.DictReader(d_labitems_file)
        for row in d_labitems_dictreader:
            d_labitems[row['ITEMID']] = {}
            d_labitems[row['ITEMID']]['ITEMID'] = row['ITEMID']
            d_labitems[row['ITEMID']]['LABEL'] = row['LABEL']
            d_labitems[row['ITEMID']]['FLUID'] = row['FLUID']
            d_labitems[row['ITEMID']]['CATEGORY'] = row['CATEGORY']
            d_labitems[row['ITEMID']]['LOINC_CODE'] = row['LOINC_CODE']

    # CREATING patients
    with open(patient_csv_path, 'r') as d_patients_file:
        d_items_dictreader = csv.DictReader(d_patients_file)
        for row in d_items_dictreader:
            patients[row['SUBJECT_ID']] = {}
            patients[row['SUBJECT_ID']]['SUBJECT_ID'] = row['SUBJECT_ID']
            patients[row['SUBJECT_ID']]['GENDER'] = row['GENDER']

    admissions_dict_reader = csv.DictReader(admissions_csv_file)
    print("============ Start admission reading ============")
    num_rows_processed = 0
    for admission_row in admissions_dict_reader:
        #Get date as date object
        admission_time = admission_row['ADMITTIME']
        admission_time = datetime.datetime.strptime(admission_time, "%Y-%m-%d %H:%M:%S")
        admission_json_object_path = json_files_path+str(admission_time.year)
        #Separating json files by year
        if not os.path.exists(admission_json_object_path):
            os.mkdir(admission_json_object_path)

        admission_json_object = dict()
        for field in DATA_FIELDS['ADMISSIONS']:
            admission_json_object[field.lower()] = admission_row[field]
        admission_json_object['GENDER'] = patients[admission_row['SUBJECT_ID']]['GENDER']

        for path in paths:
            data_path = mimic_data_path+path
            file_csv_path = data_path + "/{}_{}.csv".format(path, admission_row["HADM_ID"])
            fields = DATA_FIELDS[path]
            if os.path.exists(file_csv_path):
                admission_json_object[path.lower()] = []
                with open(file_csv_path, 'r') as file_csv:
                    file_csv_dictreader = csv.DictReader(file_csv)
                    for file_row in file_csv_dictreader:
                        row_new_object = dict()
                        for field in fields:
                            row_new_object[field.lower()] = file_row[field]
                        # Adding fields for d_items
                        if path in D_ITEMS_RELATION.keys():
                            for key in D_ITEMS_RELATION[path].keys():
                                if key in file_row.keys() and file_row[key] in d_items.keys():
                                    for kkey in D_ITEMS_RELATION[path][key].keys():
                                        if kkey in d_items[file_row[key]]:
                                            row_new_object[D_ITEMS_RELATION[path][key][kkey]] = d_items[file_row[key]][kkey]
                        if path in D_LABITEMS_RELATION.keys():
                            for key in D_LABITEMS_RELATION[path].keys():
                                if key in file_row.keys() and file_row[key] in d_labitems.keys():
                                    for kkey in D_LABITEMS_RELATION[path][key].keys():
                                        if kkey in d_labitems[file_row[key]]:
                                            row_new_object[D_LABITEMS_RELATION[path][key][kkey]] = d_labitems[file_row[key]][kkey]
                        admission_json_object[path.lower()].append(row_new_object)

        # pp.pprint(admission_json_object)
        with open(admission_json_object_path+'/'+admission_row['HADM_ID']+'.json', 'w') as admission_json_file:
            json.dump(admission_json_object, admission_json_file, sort_keys=True, indent=4, separators=(', ', ': '))
        num_rows_processed += 1
        if num_rows_processed % 1000 == 0:
            print("============ {} rows processed ============".format(num_rows_processed))