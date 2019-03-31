from datetime import datetime

import pandas as pd
import numpy as np
"""
This script calculates the SOFA for each icustay per hour for the whole period.
The SOFA calculation is based on https://github.com/MIT-LCP/mimic-code/blob/master/concepts/severityscores/sofa.sql
Each sql dependency to execute the sql script mentioned above was changed to reflect not only the first day, but
for the period of the icu stay.
To execute this python script, is necessary the csv of those dependencies, and the execution of the split_by_admission.py.
"""
datetime_pattern = "%Y-%m-%d %H:%M:%S"
# Using variables for the paths to the files
mimic_data_path = "/home/mattyws/Documents/mimic_data/"
csv_file_path = mimic_data_path+"csv/"
chartevents_path = mimic_data_path+'CHARTEVENTS/'
inputevents_mv_path = mimic_data_path+'INPUTEVENTS_MV/'
inputevents_cv_path = mimic_data_path+'INPUTEVENTS_CV/'
# Auxiliary variables for events ids
weights_kg_ids = [762, 763, 3723, 3580, 226512]
weights_lbs_ids = [3581]
weights_oz_ids = [3582]

vasopressor_cv_ids = [30047,30120,30044,30119,30309,30043,30307,30042,30306]
norepinephrine_cv_ids = [30047, 30120]
epinephrine_cv_ids = [30044, 30119, 30309]
dopamine_cv_ids = [30043, 30307]
dobutamine_cv_ids = [30042, 30306]

vasopressor_mv_ids = [221906,221289,221662,221653]

# Variables for the csv created by the sql dependencies
print("=== Loading csv ===")
patients = pd.read_csv(csv_file_path+'PATIENTS.csv')
bloodgasarterial = pd.read_csv(mimic_data_path+'bloodgasarterial.csv')
gcs = pd.read_csv(mimic_data_path+'gcs.csv')
labs = pd.read_csv(mimic_data_path+'labs.csv')
urine_output = pd.read_csv(mimic_data_path+'urineoutput.csv')
vitals = pd.read_csv(mimic_data_path+'vitals.csv')
echodata = pd.read_csv(mimic_data_path+'echodata.csv')
ventdurations = pd.read_csv(mimic_data_path+'ventdurations.csv')

# Reading the icustays.csv downloaded at the mimic repository
icustays = pd.read_csv(csv_file_path+'ICUSTAYS.csv')
# Loop through each icu stay
for index, icustay in icustays.iterrows():
    print(icustay[['ICUSTAY_ID', 'DBSOURCE', 'INTIME', 'OUTTIME']])
    patient = patients[patients['SUBJECT_ID'] == icustay['SUBJECT_ID']].iloc[0]
    dob_datetime = datetime.strptime(patient['DOB'], datetime_pattern)
    intime_datetime = datetime.strptime(icustay['INTIME'], datetime_pattern)
    age = intime_datetime.year - dob_datetime.year - ((intime_datetime.month, intime_datetime.day)
                                                      < (dob_datetime.month, dob_datetime.day))
    if not icustay['HADM_ID'] or age < 16:
        continue
    # Get events on chartevents that are associated to this icu stay
    try:
        icu_chartevents = pd.read_csv(chartevents_path+'CHARTEVENTS_{}.csv'.format(icustay['HADM_ID']))
    except:
        continue
    icu_chartevents = icu_chartevents[icu_chartevents['ICUSTAY_ID'] == icustay['ICUSTAY_ID']]
    # Calculate the weight for each time that appears. Convert all weight that are not in KG to KG
    # Get weight events in KG
    weights = icu_chartevents[icu_chartevents['ITEMID'].isin(weights_kg_ids)]
    # Get weights events in lb and convert
    aux_weights = icu_chartevents[icu_chartevents['ITEMID'].isin(weights_lbs_ids)]
    if len(aux_weights) != 0:
        aux_weights['VALUENUM'] = aux_weights['VALUENUM'].apply(lambda x: x * 0.45359237)
        weights = pd.concat([weights, aux_weights], ignore_index=True)
    # Get weights events in oz and convert
    aux_weights = icu_chartevents[icu_chartevents['ITEMID'].isin(weights_oz_ids)]
    if len(aux_weights) != 0:
        aux_weights['VALUENUM'] = aux_weights['VALUENUM'].apply(lambda x: x * 0.0283495231)
        weights = pd.concat([weights, aux_weights], ignore_index=True)
    weights = weights[(weights['VALUENUM'].notna()) & (weights['ERROR'] != 1)]
    # Transform datetime
    weights['CHARTTIME'] = pd.to_datetime(weights['CHARTTIME'], format=datetime_pattern)
    # Get the weight from echo data if the patient is missing the weight event
    icu_echodata = echodata[echodata['hadm_id'] == icustay['HADM_ID']]
    # Transform datetime
    icu_echodata['charttime'] = pd.to_datetime(icu_echodata['charttime'], format=datetime_pattern)
    # print(icu_echodata[['charttime', 'weight']])
    # Get vasopressor, depending if patient is in metavision or in carevue
    icu_vasopressor_events = None
    if icustay['DBSOURCE'] == 'carevue':
        # Get vasopressor events
        try:
            icu_vasopressor_events = pd.read_csv(inputevents_cv_path+'INPUTEVENTS_CV_{}.csv'.format(icustay['HADM_ID']))
        except:
            continue
        icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['ICUSTAY_ID'] == icustay['ICUSTAY_ID']]
        icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['ITEMID'].isin(vasopressor_cv_ids)]
        # Removing na rate
        icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['RATE'].notna()]
        # Transform datetime
        icu_vasopressor_events['CHARTTIME'] = pd.to_datetime(icu_vasopressor_events['CHARTTIME'], format=datetime_pattern)
        # Get patient weight for each vasopressor charttime
        aux_weights = []
        for index, vaso_event in icu_vasopressor_events.iterrows():
            if len(weights['CHARTTIME']) != 0:
                weight = min(weights['CHARTTIME'], key=lambda x: abs(x - vaso_event['CHARTTIME']))
                weight = weights[weights['CHARTTIME'] == weight].iloc[0]['VALUENUM']
            else:
                weight = min(icu_echodata['charttime'], key=lambda x: abs(x - vaso_event['CHARTTIME']))
                weight = icu_echodata[icu_echodata['charttime'] == weight].iloc[0]['weight']
            aux_weights.append(weight)
        # Transform for the ids that are not measured by the patient weight
        icu_vasopressor_events['weight'] = aux_weights
        icu_vasopressor_events['RATE'] = np.where(icu_vasopressor_events['ITEMID'] == norepinephrine_cv_ids[0],
                                                  icu_vasopressor_events['RATE'] / icu_vasopressor_events['weight'],
                                                  icu_vasopressor_events['RATE'])
        icu_vasopressor_events['RATE'] = np.where(icu_vasopressor_events['ITEMID'] == epinephrine_cv_ids[0],
                                                  icu_vasopressor_events['RATE'] / icu_vasopressor_events['weight'],
                                                  icu_vasopressor_events['RATE'])
    else:
        # Is a metavision icu stay
        try:
            icu_vasopressor_events = pd.read_csv(inputevents_cv_path+'INPUTEVENTS_MV_{}.csv'.format(icustay['HADM_ID']))
        except:
            continue
        icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['ICUSTAY_ID'] == icustay['ICUSTAY_ID']]
        icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['ITEMID'].isin(vasopressor_mv_ids)]
        # Removing na rate
        icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['RATE'].notna()]
        # Removing rewriten status
        icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['STATUSDESCRIPTION'] != 'Rewritten']
        # Transform datetime
        icu_vasopressor_events['CHARTTIME'] = pd.to_datetime(icu_vasopressor_events['CHARTTIME'],
                                                             format=datetime_pattern)
    is_ventilated = icustay['ICUSTAY_ID'] in ventdurations['icustay_id']
    icu_bloodgasarterial = bloodgasarterial[bloodgasarterial['icustay_id'] == icustay['ICUSTAY_ID']]
    icu_vitals = vitals[vitals['icustay_id'] == icustay['ICUSTAY_ID']]
    icu_gcs = gcs[gcs['icustay_id'] == icustay['ICUSTAY_ID']]
    icu_urine_output = urine_output[urine_output['icustay_id'] == icustay['ICUSTAY_ID']]
    icu_labs = labs[labs['icustay_id'] == icustay['ICUSTAY_ID']]

    # TODO : Calculating sofa score
    platelet_min = icu_labs[['charttime', 'platelet_min']].dropna()
    bilirubin_max = icu_labs[['charttime', 'bilirubin_max']].dropna()
    creatinine_max = icu_labs[['charttime', 'creatinine_max']].dropna()
    mingcs = icu_gcs[['charttime', 'mingcs']].dropna()
    pao2fio2 = icu_bloodgasarterial[['charttime', 'pao2fio2']].dropna()
    exit()
