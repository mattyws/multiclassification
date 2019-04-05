import os
from datetime import datetime, timedelta

import pandas as pd


datetime_pattern = "%Y-%m-%d %H:%M:%S"
mimic_data_path = "/home/mattyws/Documentos/mimic/data/"
sofa_scores_files_path = mimic_data_path+"sofa_scores/"

sepsis3_df_no_exclusions = pd.read_csv('sepsis3-df-no-exclusions.csv')
infected_icu = sepsis3_df_no_exclusions[sepsis3_df_no_exclusions['suspected_infection_time_poe'].notna()]
infected_icu['intime'] = pd.to_datetime(infected_icu['intime'], format=datetime_pattern)
infected_icu['outtime'] = pd.to_datetime(infected_icu['outtime'], format=datetime_pattern)
# infected_icu['suspected_infection_time_poe'] = pd.to_datetime(infected_icu['suspected_infection_time_poe'],
#                                                               format=datetime_pattern)
# infected_icu = infected_icu[infected_icu['icustay_id'] == 200003]

sepsis3_patients = pd.DataFrame([])
less_7 = 0
for index, infected_patient in infected_icu.iterrows():
    aux_patient = None
    try:
        intime = infected_patient['intime']
    except:
        continue
    outtime = infected_patient['outtime']
    infection_time = datetime.strptime(infected_patient['suspected_infection_time_poe'], datetime_pattern)
    # Get only patients which do not enter on icu with a suspicion of infection
    if infection_time < intime:
        continue
    # Get the time window that goes from 48 hours before the suspicion to 24 hours after
    if infection_time - timedelta(hours=48) < intime:
        begin_time_window = intime
    else:
        begin_time_window = infection_time - timedelta(hours=48)
    if infection_time + timedelta(hours=24) > outtime:
        end_time_window = outtime
    else:
        end_time_window = infection_time + timedelta(hours=24)
    if not os.path.exists(sofa_scores_files_path+'{}.csv'.format(infected_patient['icustay_id'])):
        continue
    # Get sofa scores
    patient_sofa_scores = pd.read_csv(sofa_scores_files_path+'{}.csv'.format(infected_patient['icustay_id']))
    patient_sofa_scores['timestep'] = pd.to_datetime(patient_sofa_scores['timestep'], format=datetime_pattern)
    patient_sofa_scores = patient_sofa_scores[(patient_sofa_scores['timestep'] >= begin_time_window )
                                              & (patient_sofa_scores['timestep'] <= end_time_window)]
    patient_sofa_scores = patient_sofa_scores.set_index('timestep').sort_index()
    # If is empty, pass this icu
    if len(patient_sofa_scores) == 0:
        continue
    # Get the closest timestamp key to the window's beginning
    closest_begin_time = min(patient_sofa_scores.index, key=lambda x: abs(x - begin_time_window))
    begin_sofa_score = patient_sofa_scores[patient_sofa_scores.index == closest_begin_time].iloc[0]
    for index, sofa_score in patient_sofa_scores.iterrows():
        # print("sofa", sofa_score,"begin", begin_sofa_score)
        if sofa_score['sofa_score']- begin_sofa_score['sofa_score'] >= 2:
            aux_patient = dict()
            aux_patient['time_sepsis'] = sofa_score.name
            aux_patient['hadm_id'] = infected_patient['hadm_id']
            aux_patient['icustay_id'] = infected_patient['icustay_id']
            aux_patient['intime'] = infected_patient['intime']
            aux_patient['outtime'] = infected_patient['outtime']
            aux_patient['suspected_infection_time_poe'] = infected_patient['suspected_infection_time_poe']
            aux_patient['dbsource'] = infected_patient['dbsource']
            aux_patient['age'] = infected_patient['age']
            aux_patient['sex'] = infected_patient['gender']
            aux_patient['ethnicity'] = infected_patient['ethnicity']
            break

    if aux_patient is not None and (aux_patient['time_sepsis'] - intime).seconds/3600 >= 7:
        print(aux_patient['icustay_id'])
        aux_patient = pd.DataFrame(aux_patient, index=[0])
        sepsis3_patients = pd.concat([sepsis3_patients, aux_patient], ignore_index=True)
    elif aux_patient is not None and (aux_patient['time_sepsis'] - intime).seconds/3600 < 7:
        less_7 += 1


print(len(sepsis3_patients))
print(less_7)
    # exit()