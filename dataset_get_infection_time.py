"""
Get the suspicion of infection time for each icustay
The suspicion of infection is given by:
    If antibiotics is prescribed first, a culture have to be taken in 72h
    If the culture is first, a antibiotic have to be administered in 24h
The events from prescription and microbiologyevents considered are the ones thar occur IN the icustay (between intime
and outtime), because MIMIC don't have events charted out of the icu context, which reflects into the inability
to calculate the SOFA score outside the icustay.
Antibiotic and routes names came from https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/abx-poe-list.sql
"""
from datetime import timedelta

import pandas as pd

import functions

parameters = functions.load_parameters_file()
# For now, load all the prescriptions and microbiologyevents csv into memory (they are small, but may be changed in
# future)
print("Loading prescriptions")
prescriptions = pd.read_csv(parameters['mimic_data_path'] + parameters['csv_files_directory'] + 'PRESCRIPTIONS.csv')
print("Loading microbiologyevents")
microbiologyevents = pd.read_csv(parameters['mimic_data_path'] + parameters['csv_files_directory']
                                 + 'MICROBIOLOGYEVENTS.csv')
print("Loading icustays")
icustays = pd.read_csv(parameters['mimic_data_path'] + parameters['csv_files_directory'] + 'ICUSTAYS.csv')

antibiotics_names = [
    'adoxa', 'ala-tet', 'alodox', 'amikacin', 'amikin', 'amoxicillin', 'clavulanate',
    'ampicillin', 'augmentin', 'avelox', 'avidoxy', 'azactam', 'azithromycin', 'aztreonam', 'axetil', 'bactocill',
    'bactrim', 'bethkis', 'biaxin', 'bicillin l-a', 'cayston', 'cefazolin', 'cedax', 'cefoxitin', 'ceftazidime',
    'cefaclor', 'cefadroxil', 'cefdinir', 'cefditoren', 'cefepime', 'cefotetan', 'cefotaxime', 'cefpodoxime',
    'cefprozil', 'ceftibuten', 'ceftin', 'cefuroxime ', 'cefuroxime', 'cephalexin', 'chloramphenicol', 'cipro',
    'ciprofloxacin', 'claforan', 'clarithromycin', 'cleocin', 'clindamycin', 'cubicin', 'dicloxacillin', 'doryx',
    'doxycycline', 'duricef', 'dynacin', 'ery-tab', 'eryped', 'eryc', 'erythrocin', 'erythromycin', 'factive',
    'flagyl', 'fortaz', 'furadantin', 'garamycin', 'gentamicin', 'kanamycin', 'keflex', 'ketek', 'levaquin',
    'levofloxacin', 'lincocin', 'macrobid', 'macrodantin', 'maxipime', 'mefoxin', 'metronidazole', 'minocin',
    'minocycline', 'monodox', 'monurol', 'morgidox', 'moxatag', 'moxifloxacin', 'myrac', 'nafcillin sodium',
    'nicazel doxy 30', 'nitrofurantoin', 'noroxin', 'ocudox', 'ofloxacin', 'omnicef', 'oracea', 'oraxyl',
    'oxacillin', 'pc pen vk', 'pce dispertab', 'panixine', 'pediazole', 'penicillin', 'periostat', 'pfizerpen',
    'piperacillin', 'tazobactam', 'primsol', 'proquin', 'raniclor', 'rifadin', 'rifampin', 'rocephin', 'smz-tmp',
    'septra', 'septra ds', 'septra', 'solodyn', 'spectracef', 'streptomycin sulfate', 'sulfadiazine',
    'sulfamethoxazole', 'trimethoprim', 'sulfatrim', 'sulfisoxazole', 'suprax', 'synercid', 'tazicef', 'tetracycline',
    'timentin', 'tobi', 'tobramycin', 'trimethoprim', 'unasyn', 'vancocin', 'vancomycin', 'vantin', 'vibativ',
    'vibra-tabs', 'vibramycin', 'zinacef', 'zithromax', 'zmax', 'zosyn', 'zyvox'
]

for index, icustay in icustays.iterrows():
    icu_prescriptions = prescriptions[prescriptions['ICUSTAY_ID'] == icustay['ICUSTAY_ID']]
    icu_prescriptions.loc[:, 'DRUG'] = icu_prescriptions['DRUG'].apply(lambda x: x.lower() if not pd.isna(x) else x)
    # Get the antibiotics prescriptions for this icu
    icu_prescriptions = icu_prescriptions[
        (icu_prescriptions['DRUG'].isin(antibiotics_names))
        | (
            (icu_prescriptions['DRUG'].str.startswith('amoxicillin'))
            & (icu_prescriptions['DRUG'].str.endswith('amoxicillin'))
        )
        & (icu_prescriptions['DRUG_TYPE'].isin(['MAIN', 'ADDITIVE']))
    ]
    # Now remove the antibiotics based on the route of administration
    icu_prescriptions = icu_prescriptions[
        ~(icu_prescriptions['ROUTE'].isin(['OU', 'OS', 'OD', 'AU', 'AS', 'AD', 'TP']))
        & ~(icu_prescriptions['ROUTE'].str.contains('eye'))
        & ~(icu_prescriptions['ROUTE'].str.contains('ear'))
        & ~(icu_prescriptions['ROUTE'].str.contains('cream'))
        & ~(icu_prescriptions['ROUTE'].str.contains('desensitization'))
        & ~(icu_prescriptions['ROUTE'].str.contains('ophth oint'))
        & ~(icu_prescriptions['ROUTE'].str.contains('gel'))
    ]
    icu_prescriptions.loc[:, 'STARTDATE'] = pd.to_datetime(icu_prescriptions['STARTDATE'],
                                                           format=parameters['date_pattern'])
    icu_prescriptions = icu_prescriptions.sort_values(by=['STARTDATE'])
    print(icu_prescriptions[['HADM_ID', 'DRUG', 'ROUTE', 'ICUSTAY_ID', 'STARTDATE', 'ENDDATE']])
    icu_microbiologyevents = microbiologyevents[
        (microbiologyevents['CHARTTIME'] >= icustay['INTIME']) & (microbiologyevents['CHARTTIME'] <= icustay['OUTTIME'])
        & (icustay['HADM_ID'] == microbiologyevents['HADM_ID'])
    ]
    icu_microbiologyevents.loc[:, 'CHARTTIME'] = pd.to_datetime(icu_microbiologyevents['CHARTTIME'],
                                                                format=parameters['datetime_pattern'])
    icu_microbiologyevents = icu_microbiologyevents.sort_values(by=['CHARTTIME'])
    print(icu_microbiologyevents[['HADM_ID', 'CHARTTIME', 'ORG_NAME']])
    infection_time_poe = None
    for index2, prescription in icu_prescriptions.iterrows():
        before_prescription = prescription['STARTDATE'] - timedelta(hours=24)
        aux_microbiologyevents = icu_microbiologyevents[
            (icu_microbiologyevents['CHARTTIME'] <= before_prescription)
        ]
        if len(aux_microbiologyevents) != 0:
            infection_time_poe = aux_microbiologyevents.iloc[0]['CHARTTIME']
            break
        after_prescription = prescription['STARTDATE'] + timedelta(hours=72)
        aux_microbiologyevents = icu_microbiologyevents[
            (icu_microbiologyevents['CHARTTIME'] >= after_prescription)
        ]
        if len(aux_microbiologyevents) != 0:
            infection_time_poe = prescription['STARTDATE']
            break
    print(infection_time_poe)
    exit()
