import os
from datetime import datetime, timedelta

import pandas as pd

import functions

parameters = functions.load_parameters_file()

datetime_pattern = parameters["datetime_pattern"]
mimic_data_path = parameters["mimic_data_path"]

infected_icu = pd.read_csv(mimic_data_path+parameters['sepsis3_df_no_exclusions'])
# Using pivoted_sofa.csv
sofa_scores = pd.read_csv(mimic_data_path+parameters['pivoted_sofa'])
sofa_scores.loc[:, 'starttime'] = pd.to_datetime(sofa_scores['starttime'], format=datetime_pattern)
sofa_scores.loc[:, 'endtime'] = pd.to_datetime(sofa_scores['endtime'], format=datetime_pattern)
# infected_icu = infected_icu[infected_icu['suspected_infection_time_poe'].notna()]
infected_icu['intime'] = pd.to_datetime(infected_icu['intime'], format=datetime_pattern)
infected_icu['outtime'] = pd.to_datetime(infected_icu['outtime'], format=datetime_pattern)
total_aleatory_patients = 20000

admissions = pd.read_csv(mimic_data_path + parameters['csv_files_directory']+'ADMISSIONS.csv')

sepsis3_patients = pd.DataFrame([])
less_7 = 0
metavision = 0
file_errors = 0
errors_metavision = 0
infected_patients = 0
healthy_patients = 0
not_infected_patients = 0
aleatory_patients = 0
for index, infected_patient in infected_icu.iterrows():
    admission = admissions[admissions['HADM_ID'] == infected_patient['hadm_id']].iloc[0]
    aux_patient = None
    try:
        intime = infected_patient['intime']
    except:
        continue
    outtime = infected_patient['outtime']
    if infected_patient['suspicion_poe']:
        infection_time = datetime.strptime(infected_patient['suspected_infection_time_poe'], datetime_pattern)
    else:
        infection_time = intime + timedelta(hours=48)
    # Get sofa scores
    try:
        patient_sofa_scores = sofa_scores[sofa_scores['icustay_id'] == infected_patient['icustay_id']]
    except:
        file_errors += 1
        errors_metavision += 1 if infected_patient['dbsource'] == 'metavision' else 0
        continue
    if patient_sofa_scores.empty:
        errors_metavision += 1 if infected_patient['dbsource'] == 'metavision' else 0
        file_errors += 1
        continue
    patient_sofa_scores['timestep'] = pd.to_datetime(patient_sofa_scores['starttime'], format=datetime_pattern)
    patient_sofa_scores = patient_sofa_scores.set_index('timestep').sort_index()
    # Get only events that occurs in before 48h up to 24h after the infection time
    patient_sofa_scores = patient_sofa_scores.truncate(before=infection_time - timedelta(hours=48))
    patient_sofa_scores = patient_sofa_scores.truncate(after=infection_time + timedelta(hours=24))
    # patient_sofa_scores = patient_sofa_scores.set_index('timestep').sort_index()
    # If is empty, pass this icu
    if len(patient_sofa_scores) == 0:
        continue
    # Get the sofa score for the beginning of the window
    begin_sofa_score = patient_sofa_scores.iloc[0]
    is_healthy = True
    for i, sofa_score in patient_sofa_scores.iterrows():
        # print("sofa", sofa_score,"begin", begin_sofa_score)
        if sofa_score['sofa_24hours'] != 0:
            is_healthy = False
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
            break

    if aux_patient is not None:
        difference = aux_patient['sofa_increasing_time_poe'] - intime
        if difference.days >= 1:
        # if difference.days > 0 or (difference.days == 0 and difference.seconds/3600 >= 7):
            if aux_patient['is_infected']:
                infected_patients += 1
                metavision += 1 if infected_patient['dbsource'] == 'metavision' else 0
            else:
                not_infected_patients += 1
            print(aux_patient['icustay_id'])
            aux_patient = pd.DataFrame(aux_patient, index=[0])
            sepsis3_patients = pd.concat([sepsis3_patients, aux_patient], ignore_index=True)

    if is_healthy:
        aux_patient = dict()
        aux_patient['is_healthy'] = is_healthy
        aux_patient['is_infected'] = infected_patient['suspicion_poe']
        aux_patient['hadm_id'] = infected_patient['hadm_id']
        aux_patient['icustay_id'] = infected_patient['icustay_id']
        aux_patient['intime'] = infected_patient['intime']
        aux_patient['outtime'] = infected_patient['outtime']
        aux_patient['dbsource'] = infected_patient['dbsource']
        aux_patient['age'] = infected_patient['age']
        aux_patient['sex'] = infected_patient['gender']
        aux_patient['ethnicity'] = infected_patient['ethnicity']
        aux_patient['class'] = "healthy"
        aux_patient = pd.DataFrame(aux_patient, index=[0])
        sepsis3_patients = pd.concat([sepsis3_patients, aux_patient], ignore_index=True)
        healthy_patients += 1
    else :
        if aux_patient is None and aleatory_patients <= total_aleatory_patients:
            aux_patient = dict()
            aux_patient['is_healthy'] = is_healthy
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
print("Healthy patients {}".format(healthy_patients))
sepsis3_patients.to_csv(parameters["dataset_file_name"])
    # exit()