"""
This script will fix the pivoted_sofa generated table (REFERENCE FOR PIVOTED SOFA QUERY ON GITHUB)
The problems is: TODO: describe problem
"""
import os

import pandas as pd

import functions

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

def get_respiration_score(pao2fio2_vent, pao2fio2_novent):
    """
    Get sofa score for respiration function
    :param pao2fio2: the value of PaO2FiO2
    :param is_vent: if the patient is on ventilation
    :return: a number between 0-4 depending on the value of PaO2FiO2
    """
    if pao2fio2_vent < 100: return 4
    elif pao2fio2_vent < 200: return 3
    elif pao2fio2_novent < 300: return 2
    elif pao2fio2_novent < 400: return 1
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
    elif (mean_bp is not None and mean_bp < 70):
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

parameters = functions.load_parameters_file()

if not os.path.exists('new_pivoted_sofa.csv'):
    print("Reading pivoted sofa")
    pivoted_sofa = pd.read_csv(parameters['pivoted_sofa'])
    variables_columns = ['icustay_id', 'hr', 'starttime', 'endtime',
                         'pao2fio2ratio_novent', 'pao2fio2ratio_vent', 'rate_epinephrine',
                         'rate_norepinephrine', 'rate_dopamine', 'rate_dobutamine', 'meanbp_min',
                         'gcs_min', 'urineoutput', 'bilirubin_max', 'creatinine_max', 'platelet_min']
    print("Filtering columns")
    pivoted_sofa = pivoted_sofa.loc[:, pivoted_sofa.columns.isin(variables_columns)]
    # Loop through sofa scores for each icustay
    new_pivoted_sofa = pd.DataFrame([])
    for icustay_id, icu_pivoted_sofa in pivoted_sofa.groupby('icustay_id'):
        icu_pivoted_sofa = icu_pivoted_sofa.ffill()
        icu_pivoted_sofa = icu_pivoted_sofa.bfill()
        new_pivoted_sofa = pd.concat([new_pivoted_sofa, icu_pivoted_sofa], ignore_index=True)
    new_pivoted_sofa.to_csv('new_pivoted_sofa.csv', index=False)
else:
    print("Reading pivoted sofa")
    new_pivoted_sofa = pd.read_csv('new_pivoted_sofa.csv')

# Calculate sofa score
print("Calculating sofa")
cardiovascular = []
coagulation = []
respiratory = []
renal = []
neurological = []
liver = []
for index, row in new_pivoted_sofa.iterrows():
    cardiovascular.append(get_cardiovascular_score(
        row['rate_dopamine'],
        row['rate_epinephrine'],
        row['rate_norepinephrine'],
        row['rate_dobutamine'],
        row['meanbp_min'],
    ))
    coagulation.append(get_coagulation_score(row['platelet_min']))
    respiratory.append(get_respiration_score(
        row['pao2fio2ratio_vent'],
        row['pao2fio2ratio_novent']
    ))
    renal.append(get_renal_score(
        row['urineoutput'],
        row['creatinine_max']
    ))
    neurological.append(get_neurological_score(row['gcs_min']))
    liver.append(get_liver_score(row['bilirubin_max']))
new_pivoted_sofa['cardiovascular'] = cardiovascular
new_pivoted_sofa['coagulation'] = coagulation
new_pivoted_sofa['respiratory'] = respiratory
new_pivoted_sofa['renal'] = renal
new_pivoted_sofa['neurological'] = neurological
new_pivoted_sofa['liver'] = liver
# Just to keep the same name as the original pivoted_sofa (scripts use this name)
new_pivoted_sofa['sofa_24hours'] = new_pivoted_sofa['cardiovascular'] + new_pivoted_sofa['coagulation'] \
                                   + new_pivoted_sofa['respiratory'] + new_pivoted_sofa['renal'] \
                                   + new_pivoted_sofa['neurological'] + new_pivoted_sofa['liver']
new_pivoted_sofa.to_csv('pivoted_sofa.csv', index=False)