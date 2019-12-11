"""
Filter selected features ids in mimic dataset and gets statistical values about the filtered features
"""
from functools import partial
from itertools import product

import pandas as pd
import numpy as np
import multiprocessing as mp
import pprint

import sys

import functions
import os

def filter_features(files_list, events_ids, dataset_filtered_files_path, manager_queue=None):
    all_ids = []
    for key in events_ids.keys():
        all_ids.extend(events_ids[key])
    data_statistic = dict()
    for f in files_list:
        if os.path.exists(dataset_filtered_files_path + os.path.basename(f)):
            if manager_queue is not None:
                manager_queue.put(f)
            continue
        patient_events = pd.read_csv(f)
        ids_in_patients = ['Unnamed: 0']
        for column in patient_events.columns:
            if "Unnamed" in column:
                continue
            column_id = int(column.split('_')[1])
            if column != 'Unnamed: 0' and column_id in all_ids:
                ids_in_patients.append(column)
            # except Exception as e:
            #     print(e)
            #     print(column)
        patient_events = patient_events[ids_in_patients]
        patient_events.loc[:, 'Unnamed: 0'] = pd.to_datetime(patient_events['Unnamed: 0'],
                                                             format=parameters['datetime_pattern'])
        patient_events = patient_events.set_index(['Unnamed: 0'])
        # Transforming and getting the meaning of equal features with different ids
        new_features = dict()
        for key in events_ids.keys():
            if key not in data_statistic.keys():
                data_statistic[key] = dict()
                data_statistic[key]["missing_patients"] = 0
                data_statistic[key]["missing_events"] = 0
                data_statistic[key]["total_events"] = 0
            new_features[key] = patient_events[
                [x for x in ids_in_patients if x != 'Unnamed: 0' and int(x.split('_')[1]) in events_ids[key]]
            ]
            if new_features[key].empty:
                new_features[key] = pd.Series(np.nan, index=patient_events.index)
                data_statistic[key]["missing_patients"] += 1
            else:
                new_features[key] = new_features[key].mean(axis=1, skipna=True)
            data_statistic[key]["missing_events"] += len([x for x in new_features[key].isna() if x is True])
            data_statistic[key]["total_events"] += len(new_features[key])
        patient_events = pd.DataFrame(new_features)

        patient_events['pulse_pressure'] = patient_events['systolic_blood_pressure'] \
            .sub(patient_events['diastolic_blood_pressure'], fill_value=0)
        patient_events['gcs_calc'] = patient_events['gcs_motor'].add(
            patient_events['gcs_eyes'].add(patient_events['gcs_verbal'], fill_value=0)
            , fill_value = 0
        )
        patient_events['gcs'] = patient_events.apply(lambda r: r['gcs_calc'] if pd.isna(r["gcs"]) else r['gcs'], axis=1)
        patient_events.loc[:, 'temperature_fahrenheit'] = (patient_events['temperature_fahrenheit'] - 32) / 1.8
        patient_events.loc[:, 'temperature_celsius'] = patient_events[['temperature_fahrenheit', 'temperature_celsius']]\
            .mean(axis=1, skipna=True)
        patient_events = patient_events.drop(
            columns=["gcs_verbal", "gcs_motor", "gcs_eyes", "gcs_calc", "temperature_fahrenheit"])
        patient_events.to_csv(dataset_filtered_files_path + os.path.basename(f))
        if manager_queue is not None:
            manager_queue.put(f)
    return data_statistic

pd.set_option('display.max_rows', None)
parameters = functions.load_parameters_file()

# sepsis_raw_merged is the merge_chartevents_labevents on raw features and renamed
dataset_merged_files_path = parameters['mimic_data_path'] + 'sepsis_raw_merged/'
dataset_filtered_files_path = parameters['mimic_data_path'] + parameters['raw_events_dirname'].format('articles')
if not os.path.exists(dataset_filtered_files_path):
    os.mkdir(dataset_filtered_files_path)

events_ids = {
    "systolic_blood_pressure" : [6, 51, 442, 455, 3313, 3315, 3317, 3321, 3323, 6701, 224167, 227243, 220050, 220179, 225309],
    "diastolic_blood_pressure" : [5364, 8364, 8368, 8440, 8441, 8502, 8503, 8504, 8506, 8507, 8555, 227242, 224643, 220051, 220180,
                                  225310],
    "mean_blood_pressure": [52, 443, 456, 3312, 6702, 6927, 220181],
    "central_venous_pressure": [716, 1103, 113, 220074],
    "temperature_celsius" : [676, 677, 223762],
    "temperature_fahrenheit" : [678, 679, 223761],
    "respiratory_rate" : [614, 615, 618, 619, 653, 1884, 8113, 3603, 224688, 224689, 224690, 220210],
    "PaO2": [490, 779, 3785, 3837, 220224],
    "FiO2": [190, 191, 3420, 3422, 1863, 2518, 2981, 7570, 223835],
    "bilirubin": [848, 5483, 5543, 4049, 3220, 5821, 1583, 5032, 5045, 4354, 225690],
    "platelets": [828, 3789, 6256, 227457],
    "creatinine": [791, 3750, 1525, 220615],
    "lactate": [818, 1531, 225668],
    "BUN": [1162, 781, 5876, 3737, 225624],
    "arterial_pH": [1126, 4753, 780, 223830],
    "WBC": [1127, 861, 4200, 1542, 220546],
    "PaCO2": [777, 778, 3784, 3835, 220235],
    "hemoglobin": [814, 220228],
    "hematocrit": [813, 3761, 226540],
    "potassium": [829, 3792, 1535, 4194, 227442, 227464],
    "glucose": [1445, 1310, 807, 811, 3744, 3745, 1529, 2338, 2416, 228388, 225664, 220621, 226537, 5431],
    "heart_rate": [211, 220045],
    "blood_oxygen_saturation": [50817],
    "gcs": [198],
    "gcs_verbal" : [723, 223900],
    "gcs_motor" : [454, 223901],
    "gcs_eyes" : [184, 220739]
}



files_list = [dataset_merged_files_path + x for x in os.listdir(dataset_merged_files_path)]
total_files = len(files_list)
with mp.Pool(processes=6) as pool:
    manager = mp.Manager()
    queue = manager.Queue()
    # Generate data to be used by the processes
    files_list = np.array_split(files_list, 10)
    # Creating a partial maintaining some arguments with fixed values
    partial_filter_features = partial(filter_features,
                                      dataset_filtered_files_path=dataset_filtered_files_path,
                                      events_ids=events_ids,
                                      manager_queue=queue)
    print("===== Filtering events =====")
    # Using starmap to pass the tuple as separated parameters to the functions
    map_obj = pool.map_async(partial_filter_features, files_list)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    results = map_obj.get()
    statistics = dict()
    for result in results:
        for key in result.keys():
            if key not in statistics.keys():
                statistics[key] = result[key]
            else:
                for key2 in statistics[key].keys():
                    statistics[key][key2] += result[key][key2]
    print()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(statistics)
    # Now loop through all created files, remove all patients with only nan's
    data_csv = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
    data_csv = data_csv.set_index(['icustay_id'])
    new_files_list = [dataset_filtered_files_path + x for x in os.listdir(dataset_filtered_files_path)]
    print("==== Removing patients with no events ====")
    consumed = 0
    icustay_to_remove = []
    for f in new_files_list:
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
        icustay_id = int(os.path.basename(f).split('.')[0])
        new_patient_events = pd.read_csv(f)
        # Don't have any events from insight, remove it from the csv and delete the file
        if new_patient_events.dropna(how='all').empty:
            os.remove(f)
            if icustay_id in data_csv.index:
                icustay_to_remove.append(icustay_id)
        else:
            # if not empty, do the na filling of values
            # new_patient_events = new_patient_events.ffill().bfill()
            new_patient_events.to_csv(f, index=False)
        consumed += 1
    print("Patients with empty events: {} ".format(len(icustay_to_remove)))
    data_csv = data_csv.drop(icustay_to_remove)
    data_csv.to_csv(parameters['mimic_data_path'] + parameters['insight_dataset_file_name'])




