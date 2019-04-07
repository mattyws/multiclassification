import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
"""
This script calculates the SOFA for each admission per hour for the whole period.
The SOFA calculation is based on https://github.com/MIT-LCP/mimic-code/blob/master/concepts/severityscores/sofa.sql
Each sql dependency to execute the sql script mentioned above was changed to reflect not only the first day, but
for the period of the icu stay.
To execute this python script, is necessary the csv of those dependencies, and the execution of the split_by_admission.py.
"""

def get_coagulation_score(platelet):
    """
    Get sofa score for coagulation function
    :param platelet: the quantity of platelet
    :return: a number between 0-4 depending on the value of platelet
    """
    if platelet is not None:
        if platelet < 20:
            return 4
        elif platelet < 50:
            return 3
        elif platelet < 100:
            return 2
        elif platelet < 150:
            return 1
    return 0

def get_respiration_score(pao2fio2, is_vent):
    """
    Get sofa score for respiration function
    :param pao2fio2: the value of PaO2FiO2
    :param is_vent: if the patient is on ventilation
    :return: a number between 0-4 depending on the value of PaO2FiO2
    """
    # if pao2fio2 is not None:
    #     if is_vent:
    #         if pao2fio2 < 100:
    #             return 4
    #         elif pao2fio2 < 200:
    #             return 3
    #     else:
    #         if pao2fio2 < 300:
    #             return 2
    #         elif pao2fio2 < 400:
    #             return 1
    if pao2fio2 is not None:
        if pao2fio2 < 100:
            return 4
        elif pao2fio2 < 220:
            return 3
        elif pao2fio2 < 300:
            return 2
        elif pao2fio2 < 400:
            return 1
    return 0

def get_liver_score(bilirubin):
    """
    Get the sofa score for liver function
    :param bilirubin: the quantity of bilirubin on blood
    :return: a number between 0-4 depending on the value of bilirubin
    """
    if bilirubin is not None:
        if bilirubin >= 12:
            return 4
        elif bilirubin >= 6:
            return 3
        elif bilirubin >= 2:
            return 2
        elif bilirubin >= 1.2:
            return 1
    return 0

def get_cardiovascular_score(rate_dopamine, rate_epinephrine, rate_norepinephrine, rate_dobutamine, mean_bp):
    """
    Get the sofa score for cardiovascular function
    :param rate_dopamine: the rate of dopamine input
    :param rate_epinephrine: the rate of epinephrine input
    :param rate_norepinephrine: the rate of norepinephrine input
    :param rate_dobutamine: the rate of dobutamine input
    :param mean_bp: the mean bp
    :return: a number between 0-4 depending on the value of the parameters
    """
    if (rate_dopamine is not None and rate_dopamine > 15) \
            or (rate_epinephrine is not None and rate_epinephrine >  0.1) \
            or (rate_norepinephrine is not None and rate_norepinephrine >  0.1):
        return 4
    elif (rate_dopamine is not None and rate_dopamine >  5) \
            or (rate_epinephrine is not None and rate_epinephrine <= 0.1) \
            or (rate_norepinephrine is not None and rate_norepinephrine <= 0.1):
        return 3
    elif (rate_dopamine is not None and rate_dopamine >  0) \
            or (rate_dobutamine is not None and rate_dobutamine > 0):
        return 2
    elif (meanbp is not None and mean_bp < 70):
        return 1
    return 0

def get_neurological_score(gcs):
    """
    Get neurological sofa score
    :param gcs: the glasgow coma score value
    :return: a number between 0-4 based on gcs score
    """
    if gcs is not None:
        if gcs >= 13 and gcs <= 14: return 1
        elif gcs >= 10 and gcs <= 12: return 2
        elif gcs >= 6 and gcs <= 9 : return 3
        elif gcs < 6: return 4
    return 0

def get_renal_score(urine_output, creatinine):
    """
    Get renal sofa score
    :param urine_output: the urine output value
    :param creatinine: the creatine value
    :return: a numbet between 0-4 based on the creatine and urine output
    """
    if (creatinine is not None and creatinine >= 5) \
            or (urine_output is not None and urine_output < 200):
        return 4
    elif (creatinine is not None and (creatinine >= 3.5 and creatinine < 5)) \
            or (urine_output is not None and urine_output < 500):
        return 3
    elif (creatinine is not None and (creatinine >= 2 and creatinine < 3.5)):
        return 2
    elif (creatinine is not None and (creatinine >= 1.2 and creatinine < 2)):
        return 1
    return 0

def get_closest_value(events, time):
    if events is None or len(events) == 0:
        return None
    event = events.truncate(after=time)
    if len(event) == 0:
        return None
    return event.iloc[-1]

datetime_pattern = "%Y-%m-%d %H:%M:%S"
# Using variables for the paths to the files
mimic_data_path = "/home/mattyws/Documents/mimic_data/"
csv_file_path = mimic_data_path+"csv/"
sofa_scores_files_path = mimic_data_path+"sofa_scores_admission/"
chartevents_path = mimic_data_path+'CHARTEVENTS/'
inputevents_mv_path = mimic_data_path+'INPUTEVENTS_MV/'
inputevents_cv_path = mimic_data_path+'INPUTEVENTS_CV/'
# Auxiliary variables for events ids
weights_kg_ids = [762, 763, 3723, 3580, 226512]
weights_lbs_ids = [3581]
weights_oz_ids = [3582]

vasopressor_ids = [30047,30120,30044,30119,30309,30043,30307,30042,30306,221906,221289,221662,221653]
norepinephrine_ids = [30047, 30120, 221906]
epinephrine_ids = [30044, 30119, 30309,221289]
dopamine_ids = [30043, 30307,221662]
dobutamine_ids = [30042, 30306,221653]
divide_rate_weight_ids = [30047, 30044]


vasopressor_cv_ids = [30047,30120,30044,30119,30309,30043,30307,30042,30306]
norepinephrine_cv_ids = [30047, 30120]
epinephrine_cv_ids = [30044, 30119, 30309]
dopamine_cv_ids = [30043, 30307]
dobutamine_cv_ids = [30042, 30306]

vasopressor_mv_ids = [221906,221289,221662,221653]
norepinephrine_mv_ids = [221906]
epinephrine_mv_ids = [221289]
dopamine_mv_ids = [221662]
dobutamine_mv_ids = [221653]


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
admissions = pd.read_csv(csv_file_path + 'ADMISSIONS.csv')
# Loop through each icu stay
for index, admission in admissions.iterrows():
    print(admission[['HADM_ID', 'ADMITTIME', 'DISCHTIME']])
    patient = patients[patients['SUBJECT_ID'] == admission['SUBJECT_ID']].iloc[0]
    dob_datetime = datetime.strptime(patient['DOB'], datetime_pattern)
    admittime_datetime = datetime.strptime(admission['ADMITTIME'], datetime_pattern)
    age = admittime_datetime.year - dob_datetime.year - ((admittime_datetime.month, admittime_datetime.day)
                                                         < (dob_datetime.month, dob_datetime.day))
    if not admission['HADM_ID'] or age < 16:
        continue
    # Get events on chartevents that are associated to this icu stay
    admit_chartevents = None
    try:
        admit_chartevents = pd.read_csv(chartevents_path + 'CHARTEVENTS_{}.csv'.format(admission['HADM_ID']))
    except:
        pass
    if admit_chartevents is not None:
        admit_chartevents = admit_chartevents[admit_chartevents['HADM_ID'] == admission['HADM_ID']]
        # Calculate the weight for each time that appears. Convert all weight that are not in KG to KG
        # Get weight events in KG
        weights = admit_chartevents[admit_chartevents['ITEMID'].isin(weights_kg_ids)]
        # Get weights events in lb and convert
        aux_weights = admit_chartevents[admit_chartevents['ITEMID'].isin(weights_lbs_ids)]
        if len(aux_weights) != 0:
            aux_weights['VALUENUM'] = aux_weights['VALUENUM'].apply(lambda x: x * 0.45359237)
            weights = pd.concat([weights, aux_weights], ignore_index=True)
        # Get weights events in oz and convert
        aux_weights = admit_chartevents[admit_chartevents['ITEMID'].isin(weights_oz_ids)]
        if len(aux_weights) != 0:
            aux_weights['VALUENUM'] = aux_weights['VALUENUM'].apply(lambda x: x * 0.0283495231)
            weights = pd.concat([weights, aux_weights], ignore_index=True)
        weights = weights[(weights['VALUENUM'].notna()) & (weights['ERROR'] != 1)]
        # Transform datetime
        weights['CHARTTIME'] = pd.to_datetime(weights['CHARTTIME'], format=datetime_pattern)
    # Get the weight from echo data if the patient is missing the weight event
    admit_echodata = echodata[echodata['hadm_id'] == admission['HADM_ID']]
    # Transform datetime
    admit_echodata['charttime'] = pd.to_datetime(admit_echodata['charttime'], format=datetime_pattern)

    # Get vasopressor, depending if patient is in metavision or in carevue
    admit_vasopressor_events = None
    if os.path.exists(inputevents_cv_path +'INPUTEVENTS_CV_{}.csv'.format(admission['HADM_ID'])):
        # Get vasopressor events
        try:
            admit_vasopressor_events = pd.read_csv(inputevents_cv_path + 'INPUTEVENTS_CV_{}.csv'.format(admission['HADM_ID']))
        except:
            pass
    elif os.path.exists(inputevents_mv_path +'INPUTEVENTS_MV_{}.csv'.format(admission['HADM_ID'])):
        # Is a metavision icu stay
        try:
            admit_vasopressor_events = pd.read_csv(inputevents_mv_path + 'INPUTEVENTS_MV_{}.csv'.format(admission['HADM_ID']))
            admit_vasopressor_events['CHARTTIME'] = admit_vasopressor_events['STORETIME']
            admit_vasopressor_events = admit_vasopressor_events.drop(columns=['STORETIME'])
        except:
            pass
    admit_norepinephrine_rate = None
    admit_epinephrine_rate = None
    admit_dopamine_rate = None
    admit_dobutamine_rate = None
    if admit_vasopressor_events is not None:
        admit_vasopressor_events = admit_vasopressor_events[admit_vasopressor_events['HADM_ID'] == admission['HADM_ID']]
        admit_vasopressor_events = admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(vasopressor_ids)]
        # Removing na rate
        admit_vasopressor_events = admit_vasopressor_events[admit_vasopressor_events['RATE'].notna()]
        # Transform datetime
        admit_vasopressor_events['CHARTTIME'] = pd.to_datetime(admit_vasopressor_events['CHARTTIME'], format=datetime_pattern)
        # Get patient weight for each vasopressor charttime
        aux_weights = []
        for index, vaso_event in admit_vasopressor_events.iterrows():
            if len(weights['CHARTTIME']) != 0:
                weight = min(weights['CHARTTIME'], key=lambda x: abs(x - vaso_event['CHARTTIME']))
                weight = weights[weights['CHARTTIME'] == weight].iloc[0]['VALUENUM']
            else:
                if len(admit_echodata) != 0:
                    weight = min(admit_echodata['charttime'], key=lambda x: abs(x - vaso_event['CHARTTIME']))
                    weight = admit_echodata[admit_echodata['charttime'] == weight].iloc[0]['weight']
                else:
                    weight = np.nan
            aux_weights.append(weight)
        # Transform for the ids that are not measured by the patient weight
        admit_vasopressor_events['weight'] = aux_weights
        # Removing events that need to be divided by weight but weight is equal no nan
        admit_vasopressor_events = admit_vasopressor_events[~(admit_vasopressor_events['ITEMID'].isin(divide_rate_weight_ids))
                                                            | (admit_vasopressor_events['weight'].notna())]
        admit_vasopressor_events['RATE'] = np.where(admit_vasopressor_events['ITEMID'].isin(divide_rate_weight_ids),
                                                    admit_vasopressor_events['RATE'] / admit_vasopressor_events['weight'],
                                                    admit_vasopressor_events['RATE'])
        admit_norepinephrine_rate = admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(norepinephrine_ids)][
            ['CHARTTIME', 'RATE']]
        admit_norepinephrine_rate['CHARTTIME'] = pd.to_datetime(admit_norepinephrine_rate['CHARTTIME'],
                                                                format=datetime_pattern)
        admit_norepinephrine_rate = admit_norepinephrine_rate.set_index('CHARTTIME').sort_index()

        admit_epinephrine_rate = admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(epinephrine_ids)][
            ['CHARTTIME', 'RATE']]
        admit_epinephrine_rate['CHARTTIME'] = pd.to_datetime(admit_epinephrine_rate['CHARTTIME'], format=datetime_pattern)
        admit_epinephrine_rate = admit_epinephrine_rate.set_index('CHARTTIME').sort_index()

        admit_dopamine_rate = admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(dopamine_ids)][
            ['CHARTTIME', 'RATE']]
        admit_dopamine_rate['CHARTTIME'] = pd.to_datetime(admit_dopamine_rate['CHARTTIME'], format=datetime_pattern)
        admit_dopamine_rate = admit_dopamine_rate.set_index('CHARTTIME').sort_index()

        admit_dobutamine_rate = admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(dobutamine_ids)][
            ['CHARTTIME', 'RATE']]
        admit_dobutamine_rate['CHARTTIME'] = pd.to_datetime(admit_dobutamine_rate['CHARTTIME'], format=datetime_pattern)
        admit_dobutamine_rate = admit_dobutamine_rate.set_index('CHARTTIME').sort_index()

    admit_ventdurations = ventdurations[ventdurations['hadm_id'] == admission['HADM_ID']]
    admit_ventdurations['starttime'] = pd.to_datetime(admit_ventdurations['starttime'], format=datetime_pattern)
    admit_ventdurations['endtime'] = pd.to_datetime(admit_ventdurations['endtime'], format=datetime_pattern)

    admit_bloodgasarterial = bloodgasarterial[bloodgasarterial['hadm_id'] == admission['HADM_ID']]
    admit_vitals = vitals[vitals['hadm_id'] == admission['HADM_ID']]
    admit_gcs = gcs[gcs['hadm_id'] == admission['HADM_ID']]
    admit_labs = labs[labs['hadm_id'] == admission['HADM_ID']]

    # Get variables to calculate the SOFA score
    admit_platelet_min = admit_labs[['charttime', 'platelet_min']].dropna()
    admit_platelet_min['charttime'] = pd.to_datetime(admit_platelet_min['charttime'], format=datetime_pattern)
    admit_platelet_min = admit_platelet_min.set_index('charttime').sort_index()

    admit_urine_output = urine_output[urine_output['hadm_id'] == admission['HADM_ID']][['charttime', 'urineoutput']]
    admit_urine_output['charttime'] = pd.to_datetime(admit_urine_output['charttime'], format=datetime_pattern)
    admit_urine_output = admit_urine_output.set_index('charttime').sort_index()

    admit_bilirubin_max = admit_labs[['charttime', 'bilirubin_max']].dropna()
    admit_bilirubin_max['charttime'] = pd.to_datetime(admit_bilirubin_max['charttime'], format=datetime_pattern)
    admit_bilirubin_max = admit_bilirubin_max.set_index('charttime').sort_index()

    admit_creatinine_max = admit_labs[['charttime', 'creatinine_max']].dropna()
    admit_creatinine_max['charttime'] = pd.to_datetime(admit_creatinine_max['charttime'], format=datetime_pattern)
    admit_creatinine_max = admit_creatinine_max.set_index('charttime').sort_index()

    admit_mingcs = admit_gcs[['charttime', 'mingcs']].dropna()
    admit_mingcs['charttime'] = pd.to_datetime(admit_mingcs['charttime'], format=datetime_pattern)
    admit_mingcs = admit_mingcs.set_index('charttime').sort_index()

    admit_pao2fio2 = admit_bloodgasarterial[['charttime', 'pao2fio2']].dropna()
    admit_pao2fio2['charttime'] = pd.to_datetime(admit_pao2fio2['charttime'], format=datetime_pattern)
    admit_pao2fio2 = admit_pao2fio2.set_index('charttime').sort_index()

    admit_meanbp = admit_vitals[['charttime', 'meanbp_min']].dropna()
    admit_meanbp['charttime'] = pd.to_datetime(admit_meanbp['charttime'], format=datetime_pattern)
    admit_meanbp = admit_meanbp.set_index('charttime').sort_index()

    # Loop from the intime of the icustay to the outtime, adding 1 hour each step
    timestep = datetime.strptime(admission['ADMITTIME'], datetime_pattern)
    try:
        dischtime_datetime = datetime.strptime(admission['DISCHTIME'], datetime_pattern)
    except:
        print("Admission do not have dischtime, ignore this stay", admission['DISCHTIME'])
        continue
    admit_sofa_scores = pd.DataFrame([])
    print("=== Calculating Sofa ===")
    while timestep <= dischtime_datetime:
        timestep_sofa_scores = dict()
        # Get values from variable based on the timestep
        platelet = get_closest_value(admit_platelet_min, timestep)
        platelet = platelet['platelet_min'] if platelet is not None else None
        # print("Platelet", platelet)

        pao2fio2 = get_closest_value(admit_pao2fio2, timestep)
        pao2fio2 = pao2fio2['pao2fio2'] if pao2fio2 is not None else None
        # print("PaO2FiO2", pao2fio2)
        # print("Is ventilated", is_ventilated)

        bilirubin = get_closest_value(admit_bilirubin_max, timestep)
        bilirubin = bilirubin['bilirubin_max'] if bilirubin is not None else None
        # print("Bilirubin", bilirubin)

        creatinine = get_closest_value(admit_creatinine_max, timestep)
        creatinine = creatinine['creatinine_max'] if creatinine is not None else None
        # print("Creatinine", creatinine)

        mingcs = get_closest_value(admit_mingcs, timestep)
        mingcs = mingcs['mingcs'] if mingcs is not None else None
        # print("MinGCS", mingcs)

        urineoutput = get_closest_value(admit_urine_output, timestep)
        urineoutput = urineoutput['urineoutput'] if urineoutput is not None else None
        # print("UrineOutput", urineoutput)

        meanbp = get_closest_value(admit_meanbp, timestep)
        meanbp = meanbp['meanbp_min'] if meanbp is not None else None
        # print("MeanBP", meanbp)

        norepinephrine_rate = get_closest_value(admit_norepinephrine_rate, timestep)
        norepinephrine_rate = norepinephrine_rate['RATE'] if norepinephrine_rate is not None else None
        # print("Norepinephrine_Rate", norepinephrine_rate)

        epinephrine_rate = get_closest_value(admit_epinephrine_rate, timestep)
        epinephrine_rate = epinephrine_rate['RATE'] if epinephrine_rate is not None else None
        # print("Epinephrine_Rate", epinephrine_rate)

        dopamine_rate = get_closest_value(admit_dopamine_rate, timestep)
        dopamine_rate = dopamine_rate['RATE'] if dopamine_rate is not None else None
        # print("Dopamine_Rate", dopamine_rate)

        dobutamine_rate = get_closest_value(admit_dobutamine_rate, timestep)
        dobutamine_rate = dobutamine_rate['RATE'] if dobutamine_rate is not None else None
        # print("Dobutamine_Rate", dobutamine_rate)

        # Get if is ventilated at this timestamp
        is_ventilated = False
        for index, ventduration in admit_ventdurations.iterrows():
            if is_ventilated:
                break
            is_ventilated = (timestep >= ventduration['starttime']) and (timestep <= ventduration['endtime'])

        timestep_sofa_scores['coagulation_score'] = get_coagulation_score(platelet)
        timestep_sofa_scores['respiration_score'] = get_respiration_score(pao2fio2, is_ventilated)
        timestep_sofa_scores['liver_score'] = get_liver_score(bilirubin)
        timestep_sofa_scores['cardiovascular_score'] = get_cardiovascular_score(dopamine_rate,
                                                                                epinephrine_rate,
                                                                                norepinephrine_rate,
                                                                                dobutamine_rate,
                                                                                meanbp)
        timestep_sofa_scores['neurological_score'] = get_neurological_score(mingcs)
        timestep_sofa_scores['renal_score'] = get_renal_score(urineoutput, creatinine)
        timestep_sofa_scores['sofa_score'] = sum([timestep_sofa_scores[key] for key in timestep_sofa_scores.keys()])
        timestep_sofa_scores['timestep'] = timestep
        timestep_sofa_scores = pd.DataFrame(timestep_sofa_scores, index=[0])
        # print(timestep_sofa_scores)
        admit_sofa_scores = pd.concat([admit_sofa_scores, timestep_sofa_scores], ignore_index=True)
        timestep += timedelta(hours=1)
    # print(icu_sofa_scores[['timestep', 'sofa_score']])
    admit_sofa_scores.to_csv(sofa_scores_files_path + '{}.csv'.format(admission['HADM_ID']))
    # exit()
