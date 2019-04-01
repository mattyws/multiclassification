from datetime import datetime, timedelta

import pandas as pd
import numpy as np
"""
This script calculates the SOFA for each icustay per hour for the whole period.
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
    if is_vent:
        if pao2fio2 < 100:
            return 4
        elif pao2fio2 < 200:
            return 3
    else:
        if pao2fio2 < 300:
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
    if rate_dopamine > 15 or rate_epinephrine >  0.1 or rate_norepinephrine >  0.1:
        return 4
    elif rate_dopamine >  5 or rate_epinephrine <= 0.1 or rate_norepinephrine <= 0.1:
        return 3
    elif rate_dopamine >  0 or rate_dobutamine > 0:
        return 2
    elif mean_bp < 70:
        return 1
    return 0

def get_neurological_score(gcs):
    """
    Get neurological sofa score
    :param gcs: the glasgow coma score value
    :return: a number between 0-4 based on gcs score
    """
    if gcs >= 13 and gcs <= 14: return 1
    elif gcs >= 10 and gcs <= 12: return 2
    elif gcs >= 6 and gcs <= 0 : return 3
    elif gcs < 6: return 4
    return 0

def get_renal_score(urine_output, creatine):
    """
    Get renal sofa score
    :param urine_output: the urine output value
    :param creatine: the creatine value
    :return: a numbet between 0-4 based on the creatine and urine output
    """
    if creatine >= 5 or urine_output < 200: return 4
    elif (creatine >= 3.5 and creatine < 5) or urine_output < 500: return 3
    elif creatine >= 2 and creatine < 3.5: return 2
    elif creatine >= 1.2 and creatine < 2: return 1
    return 0

def get_closest_value(events, time):
    if len(events) == 0:
        return None
    event = events.truncate(after=time)
    if len(event) == 0:
        return None
    return event.iloc[-1]

datetime_pattern = "%Y-%m-%d %H:%M:%S"
# Using variables for the paths to the files
mimic_data_path = "/home/mattyws/Documentos/mimic/data/"
csv_file_path = mimic_data_path+"csv/"
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
            pass
    else:
        # Is a metavision icu stay
        try:
            icu_vasopressor_events = pd.read_csv(inputevents_cv_path+'INPUTEVENTS_MV_{}.csv'.format(icustay['HADM_ID']))
        except:
            pass
    icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['ICUSTAY_ID'] == icustay['ICUSTAY_ID']]
    icu_vasopressor_events = icu_vasopressor_events[icu_vasopressor_events['ITEMID'].isin(vasopressor_ids)]
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
    icu_vasopressor_events['RATE'] = np.where(icu_vasopressor_events['ITEMID'].isin(divide_rate_weight_ids),
                                              icu_vasopressor_events['RATE'] / icu_vasopressor_events['weight'],
                                              icu_vasopressor_events['RATE'])

    is_ventilated = icustay['ICUSTAY_ID'] in ventdurations['icustay_id']
    icu_bloodgasarterial = bloodgasarterial[bloodgasarterial['icustay_id'] == icustay['ICUSTAY_ID']]
    icu_vitals = vitals[vitals['icustay_id'] == icustay['ICUSTAY_ID']]
    icu_gcs = gcs[gcs['icustay_id'] == icustay['ICUSTAY_ID']]
    icu_labs = labs[labs['icustay_id'] == icustay['ICUSTAY_ID']]

    # Get variables to calculate the SOFA score
    icu_platelet_min = icu_labs[['charttime', 'platelet_min']].dropna()
    icu_platelet_min['charttime'] = pd.to_datetime(icu_platelet_min['charttime'], format=datetime_pattern)
    icu_platelet_min = icu_platelet_min.set_index('charttime')

    icu_urine_output = urine_output[urine_output['icustay_id'] == icustay['ICUSTAY_ID']][['charttime', 'urineoutput']]
    icu_urine_output['charttime'] = pd.to_datetime(icu_urine_output['charttime'], format=datetime_pattern)
    icu_urine_output = icu_urine_output.set_index('charttime')

    icu_bilirubin_max = icu_labs[['charttime', 'bilirubin_max']].dropna()
    icu_bilirubin_max['charttime'] = pd.to_datetime(icu_bilirubin_max['charttime'], format=datetime_pattern)
    icu_bilirubin_max = icu_bilirubin_max.set_index('charttime')

    icu_creatinine_max = icu_labs[['charttime', 'creatinine_max']].dropna()
    icu_creatinine_max['charttime'] = pd.to_datetime(icu_creatinine_max['charttime'], format=datetime_pattern)
    icu_creatinine_max = icu_creatinine_max.set_index('charttime')

    icu_mingcs = icu_gcs[['charttime', 'mingcs']].dropna()
    icu_mingcs['charttime'] = pd.to_datetime(icu_mingcs['charttime'], format=datetime_pattern)
    icu_mingcs = icu_mingcs.set_index('charttime')

    icu_pao2fio2 = icu_bloodgasarterial[['charttime', 'pao2fio2']].dropna()
    icu_pao2fio2['charttime'] = pd.to_datetime(icu_pao2fio2['charttime'], format=datetime_pattern)
    icu_pao2fio2 = icu_pao2fio2.set_index('charttime')

    icu_meanbp = icu_vitals[['charttime', 'meanbp_min']].dropna()
    icu_meanbp['charttime'] = pd.to_datetime(icu_meanbp['charttime'], format=datetime_pattern)
    icu_meanbp = icu_meanbp.set_index('charttime')

    icu_norepinephrine_rate = icu_vasopressor_events[icu_vasopressor_events['ITEMID'].isin(norepinephrine_ids)][
        ['CHARTTIME', 'RATE']]
    icu_norepinephrine_rate['CHARTTIME'] = pd.to_datetime(icu_norepinephrine_rate['CHARTTIME'], format=datetime_pattern)
    icu_norepinephrine_rate = icu_norepinephrine_rate.set_index('CHARTTIME')

    icu_epinephrine_rate = icu_vasopressor_events[icu_vasopressor_events['ITEMID'].isin(epinephrine_ids)][
        ['CHARTTIME', 'RATE']]
    icu_epinephrine_rate['CHARTTIME'] = pd.to_datetime(icu_epinephrine_rate['CHARTTIME'], format=datetime_pattern)
    icu_epinephrine_rate = icu_epinephrine_rate.set_index('CHARTTIME')

    icu_dopamine_rate = icu_vasopressor_events[icu_vasopressor_events['ITEMID'].isin(dopamine_ids)][
        ['CHARTTIME', 'RATE']]
    icu_dopamine_rate['CHARTTIME'] = pd.to_datetime(icu_dopamine_rate['CHARTTIME'], format=datetime_pattern)
    icu_dopamine_rate = icu_dopamine_rate.set_index('CHARTTIME')

    icu_dobutamine_rate = icu_vasopressor_events[icu_vasopressor_events['ITEMID'].isin(dobutamine_ids)][
        ['CHARTTIME', 'RATE']]
    icu_dobutamine_rate['CHARTTIME'] = pd.to_datetime(icu_dobutamine_rate['CHARTTIME'], format=datetime_pattern)
    icu_dobutamine_rate = icu_dobutamine_rate.set_index('CHARTTIME')

    # Loop from the intime of the icustay to the outtime, adding 1 hour each step
    timestep = datetime.strptime(icustay['INTIME'], datetime_pattern)
    outtime_datetime = datetime.strptime(icustay['OUTTIME'], datetime_pattern)
    icu_sofa_scores = pd.DataFrame({})
    print("=== Calculating Sofa ===")
    while timestep <= outtime_datetime:
        print(timestep)
        timestep_sofa_scores = dict()
        # Get values from variable based on the timestep
        platelet = get_closest_value(icu_platelet_min, timestep)
        pao2fio2 = get_closest_value(icu_pao2fio2, timestep)
        bilirubin = get_closest_value(icu_bilirubin_max, timestep)
        creatinine = get_closest_value(icu_creatinine_max, timestep)
        mingcs = get_closest_value(icu_mingcs, timestep)
        urineoutput = get_closest_value(icu_urine_output, timestep)
        meanbp = get_closest_value(icu_meanbp, timestep)
        norepinephrine_rate = get_closest_value(icu_norepinephrine_rate, timestep)
        epinephrine_rate = get_closest_value(icu_epinephrine_rate, timestep)
        dopamine_rate = get_closest_value(icu_dopamine_rate, timestep)
        dobutamine_rate = get_closest_value(icu_dobutamine_rate, timestep)


        timestep_sofa_scores['coagulation_score'] = get_coagulation_score(platelet['platelet_min'])
        timestep_sofa_scores['respiration_score'] = get_respiration_score(pao2fio2['pao2fio2'], is_ventilated)
        timestep_sofa_scores['liver_score'] = get_liver_score(bilirubin['bilirubin_max'])
        timestep_sofa_scores['cardiovascular_score'] = get_cardiovascular_score(dopamine_rate['RATE'],
                                                                                epinephrine_rate['RATE'],
                                                                                norepinephrine_rate['RATE'],
                                                                                dobutamine_rate['RATE'],
                                                                                meanbp['meanbp_min'])
        timestep_sofa_scores['neurological_score'] = get_neurological_score(mingcs['mingcs'])
        timestep_sofa_scores['renal_score'] = get_renal_score(urineoutput['urineoutput'], creatinine['creatinine_max'])
        timestep_sofa_scores['sofa_score'] = sum([timestep_sofa_scores[key] for key in timestep_sofa_scores.keys()])
        print(timestep_sofa_scores)
        timestep += timedelta(hours=1)
        break


    exit()
