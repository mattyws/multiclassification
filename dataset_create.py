"""
Creates the database following the criterias:
# TODO: list the criterias
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import logging

import functions

logging.basicConfig(level=logging.INFO)

parameters = functions.load_parameters_file()

datetime_pattern = parameters["datetime_pattern"]
mimic_data_path = parameters["mimic_data_path"]

"""
The reason that we test if the dataset is already created is:
    Separate the logic of patients exclusion criteria from the filtering criteria, so it's easier to undestand the logic for both
    And honestly this script was executed and the file already exists 
"""
if not os.path.exists(mimic_data_path + parameters["raw_dataset_file_name"]):
    print("===== Creating dataset =====")
    print("Loading icustays")
    icustays = pd.read_csv(mimic_data_path + parameters['csv_files_directory'] + 'ICUSTAYS.csv')
    icustays.loc[:, 'hadm_id'] = icustays['HADM_ID']
    icustays.loc[:, 'icustay_id'] = icustays['ICUSTAY_ID']
    icustays = icustays[['icustay_id', 'hadm_id']]

    print("Loading sepsis3-df")
    infected_icu = pd.read_csv(mimic_data_path + parameters['sepsis3_df_no_exclusions'])

    print("Loading pivoted_sofa")
    # Using pivoted_sofa.csv
    sofa_scores = pd.read_csv(mimic_data_path + parameters['pivoted_sofa'])
    sofa_scores.loc[:, 'starttime'] = pd.to_datetime(sofa_scores['starttime'], format=datetime_pattern)
    sofa_scores.loc[:, 'endtime'] = pd.to_datetime(sofa_scores['endtime'], format=datetime_pattern)
    sofa_scores = pd.merge(sofa_scores, icustays, how="inner", on=["icustay_id"])
    sofa_scores = sofa_scores.sort_values(by=['hadm_id', 'icustay_id', 'starttime'])
    # infected_icu = infected_icu[infected_icu['suspected_infection_time_poe'].notna()]
    infected_icu.loc[:, 'intime'] = pd.to_datetime(infected_icu['intime'], format=datetime_pattern)
    infected_icu.loc[:, 'outtime'] = pd.to_datetime(infected_icu['outtime'], format=datetime_pattern)
    infected_icu.loc[:, 'suspected_infection_time_poe'] = pd.to_datetime(infected_icu['suspected_infection_time_poe'],
                                                                         format=datetime_pattern)
    sepsis3_patients = pd.DataFrame([])
    less_7 = 0
    metavision = 0
    file_errors = 0
    errors_metavision = 0
    infected_patients = 0
    not_infected_patients = 0
    aleatory_patients = 0
    for index, infected_patient in infected_icu.iterrows():
        if infected_patient['age'] < 15 or infected_patient['age'] > 80:
            print("Patient age do not meet criteria")
            continue
        print("==== {} ====".format(infected_patient['icustay_id']))
        aux_patient = None
        try:
            intime = infected_patient['intime']
        except:
            continue
        outtime = infected_patient['outtime']
        logging.debug("{} - {}".format(intime, outtime))
        # If the patient is with a suspicion of infection, use the time of suspicion, and if isn't with the suspicion
        # use 48h after the intime, because the window for the change on sofa that is been used is 48h before and 24h after
        if infected_patient['suspicion_poe']:
            # if infected_patient['suspected_infection_time_poe'] > infected_patient['intime']:
            infection_time = infected_patient['suspected_infection_time_poe']
            # else:
            #     infection_time = infected_patient['intime']
        else:
            infection_time = intime + timedelta(hours=48)
        # if infection_time < intime:
        #     print("Infection  time is lower than the intime")
        #     continue
        logging.debug("Infection time: {}".format(infection_time))
        # Get sofa scores
        try:
            patient_sofa_scores = sofa_scores[sofa_scores['hadm_id'] == infected_patient['hadm_id']]
        except:
            file_errors += 1
            errors_metavision += 1 if infected_patient['dbsource'] == 'metavision' else 0
            continue
        if patient_sofa_scores.empty:
            errors_metavision += 1 if infected_patient['dbsource'] == 'metavision' else 0
            file_errors += 1
            continue
        patient_sofa_scores.loc[:, 'timestep'] = pd.to_datetime(patient_sofa_scores['starttime'], format=datetime_pattern)
        patient_sofa_scores = patient_sofa_scores.set_index('timestep').sort_index()
        # Get only events that occurs in before 48h up to 24h after the infection time
        # The before is truncated by data availability, and after is truncated by the icu outtime
        before_truncate = infection_time - timedelta(hours=48)
        after_truncate = infection_time + timedelta(hours=24)
        if infected_patient['outtime'] < after_truncate:
            after_truncate = infected_patient['outtime']
        logging.debug("{} - {}".format(before_truncate, after_truncate))
        patient_sofa_scores = patient_sofa_scores.truncate(before=before_truncate)
        patient_sofa_scores = patient_sofa_scores.truncate(after=after_truncate)
        logging.debug(patient_sofa_scores.index)
        # If is empty, pass this icu
        if len(patient_sofa_scores) == 0:
            continue
        # Get the sofa score for the beginning of the window
        begin_sofa_score = patient_sofa_scores.iloc[0]
        for i, sofa_score in patient_sofa_scores.iterrows():
            if sofa_score['sofa_24hours'] - begin_sofa_score['sofa_24hours'] >= 2:
                aux_patient = dict()
                aux_patient['is_infected'] = infected_patient['suspicion_poe']
                aux_patient['sofa_increasing_time_poe'] = sofa_score.name
                aux_patient['hadm_id'] = infected_patient['hadm_id']
                aux_patient['icustay_id'] = infected_patient['icustay_id']
                aux_patient['intime'] = infected_patient['intime']
                aux_patient['outtime'] = infected_patient['outtime']
                aux_patient['suspected_infection_time_poe'] = infected_patient['suspected_infection_time_poe']
                aux_patient['dbsource'] = infected_patient['dbsource']
                aux_patient['age'] = infected_patient['age']
                aux_patient['sex'] = infected_patient['gender']
                aux_patient['ethnicity'] = infected_patient['ethnicity']
                aux_patient['class'] = "sepsis" if infected_patient['suspicion_poe'] else "no_infection"
                aux_patient['begin_sofa'] = begin_sofa_score['sofa_24hours']
                aux_patient['begin_window_time_poe'] = begin_sofa_score.name
                aux_patient['sofa_on_increasing_time'] = sofa_score['sofa_24hours']
                break

        if aux_patient is not None:
            print(aux_patient['icustay_id'])
            if aux_patient['is_infected']:
                infected_patients += 1
                metavision += 1 if infected_patient['dbsource'] == 'metavision' else 0
            else:
                not_infected_patients += 1
            aux_patient = pd.DataFrame(aux_patient, index=[0])
            logging.debug(aux_patient['class'].values)
            logging.debug("Sofa begining window time: {} - Sofa = {}".format(aux_patient['begin_window_time_poe'].values, aux_patient['begin_sofa'].values))
            logging.debug("Sofa incresing time: {} - Sofa = {}".format(aux_patient['sofa_increasing_time_poe'].values, aux_patient['sofa_on_increasing_time'].values))
            sepsis3_patients = pd.concat([sepsis3_patients, aux_patient], ignore_index=True)

        if aux_patient is None:
            aux_patient = dict()
            aux_patient['is_infected'] = infected_patient['suspicion_poe']
            aux_patient['hadm_id'] = infected_patient['hadm_id']
            aux_patient['icustay_id'] = infected_patient['icustay_id']
            aux_patient['intime'] = infected_patient['intime']
            aux_patient['outtime'] = infected_patient['outtime']
            aux_patient['dbsource'] = infected_patient['dbsource']
            aux_patient['suspected_infection_time_poe'] = infected_patient['suspected_infection_time_poe']
            aux_patient['age'] = infected_patient['age']
            aux_patient['sex'] = infected_patient['gender']
            aux_patient['ethnicity'] = infected_patient['ethnicity']
            aux_patient['class'] = "None"
            aux_patient = pd.DataFrame(aux_patient, index=[0])
            sepsis3_patients = pd.concat([sepsis3_patients, aux_patient], ignore_index=True)
            aleatory_patients += 1
    print(len(sepsis3_patients))
    print("metavision", metavision)
    print('file errors', file_errors)
    print("errors metavision", errors_metavision)
    print("Infected patients {}".format(infected_patients))
    print("Not Infected patients {}".format(not_infected_patients))
    sepsis3_patients.to_csv(mimic_data_path + parameters["raw_dataset_file_name"])
else:
    sepsis3_patients = pd.read_csv(mimic_data_path + parameters['raw_dataset_file_name'])
    sepsis3_patients.loc[:, 'sofa_increasing_time_poe'] = pd.to_datetime(sepsis3_patients['sofa_increasing_time_poe'],
                                                                         format=datetime_pattern)
    sepsis3_patients.loc[:, 'intime'] = pd.to_datetime(sepsis3_patients['intime'],
                                                                         format=datetime_pattern)

print("===== Removing patients =====")
icustays_to_remove = []
for index, patient in sepsis3_patients.iterrows():
    if pd.isna(patient['sofa_increasing_time_poe']):
        continue
    difference = patient['sofa_increasing_time_poe'] - patient['intime']
    days = difference.days
    hours = difference.seconds/3600
    if (days == 0 and hours < 7) or (days < 0) \
            or ( (days*24) + hours > 500 ):
        icustays_to_remove.append(patient['icustay_id'])
removed_patients = sepsis3_patients[sepsis3_patients['icustay_id'].isin(icustays_to_remove)]
sepsis3_patients = sepsis3_patients[~(sepsis3_patients['icustay_id'].isin(icustays_to_remove))]
removed_patients.to_csv(mimic_data_path + 'removed_patients_dataset_create.csv')
sepsis3_patients.to_csv(mimic_data_path + parameters['dataset_file_name'])
print(sepsis3_patients['class'].value_counts())
