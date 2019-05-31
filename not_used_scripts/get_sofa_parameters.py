from datetime import timedelta

import numpy as np
import pandas as pd

datetime_pattern = "%Y-%m-%d %H:%M:%S"

# def get_pao2fio2_events(labevents, chartevents)


def get_gcs_events(chartevents, admittime, dischtime, debug=False, hadm_id=0):
    GCS_ids = [184, 454, 723, 223900, 223901, 220739]
    GCSVerbal_ids = [723, 223900]
    GCSMotor_ids = [454, 223901]
    GCSEyes_ids = [184, 220739]

    gcsverbal_events = chartevents[(chartevents['ITEMID'].isin(GCSVerbal_ids)) & (chartevents['ERROR'] != 1)]
    # These values indicates that the patient is in endotrach/vent
    if len(gcsverbal_events) != 0:
        gcsverbal_events.loc[:, 'VALUENUM'] = np.where(gcsverbal_events['VALUE'] == '1.0 ET/Trach', 0,
                                                   gcsverbal_events['VALUENUM'])
        gcsverbal_events.loc[:, 'VALUENUM'] = np.where(gcsverbal_events['VALUE'] == 'No Response-ETT', 0,
                                                   gcsverbal_events['VALUENUM'])
        gcsverbal_events.loc[:, 'CHARTTIME'] = pd.to_datetime(gcsverbal_events['CHARTTIME'], format=datetime_pattern)

    gcsmotor_events = chartevents[(chartevents['ITEMID'].isin(GCSMotor_ids)) & (chartevents['ERROR'] != 1)]
    if len(gcsmotor_events) != 0:
        gcsmotor_events.loc[:, 'CHARTTIME'] = pd.to_datetime(gcsmotor_events['CHARTTIME'], format=datetime_pattern)

    gcseyes_events = chartevents[(chartevents['ITEMID'].isin(GCSEyes_ids)) & (chartevents['ERROR'] != 1)]
    if len(gcseyes_events) != 0:
        gcseyes_events.loc[:, 'CHARTTIME'] = pd.to_datetime(gcseyes_events['CHARTTIME'], format=datetime_pattern)

    timestep = admittime
    gcs_values_hourly = pd.DataFrame([])
    while timestep <= dischtime:
        timestep_gcs = dict()
        timestep_gcs['timestep'] = timestep

        closest_verbal_event = []
        if len(gcsverbal_events) != 0:
            closest_verbal_event = gcsverbal_events[gcsverbal_events['CHARTTIME'] ==
                                                    min(gcsverbal_events['CHARTTIME'], key=lambda x: abs(x - timestep))]

        closest_motor_event = []
        if len(gcsmotor_events) != 0:
            closest_motor_event = gcsmotor_events[gcsmotor_events['CHARTTIME'] ==
                                                 min(gcsmotor_events['CHARTTIME'], key=lambda x: abs(x - timestep))]

        closest_eyes_event = []
        if len(gcseyes_events) != 0:
            closest_eyes_event = gcseyes_events[gcseyes_events['CHARTTIME'] ==
                                                min(gcseyes_events['CHARTTIME'], key=lambda x: abs(x - timestep))]

        timestep_gcs['motor'] = max(closest_motor_event['VALUENUM']) if len(closest_motor_event) != 0 else 6
        timestep_gcs['verbal'] = max(closest_verbal_event['VALUENUM']) if len(closest_verbal_event) != 0 else 5
        timestep_gcs['eyes'] = max(closest_eyes_event['VALUENUM']) if len(closest_eyes_event) != 0 else 4
        if timestep_gcs['verbal'] is None or (timestep_gcs['verbal'] is not None and timestep_gcs['verbal'] == 0):
            timestep_gcs['gcs'] = 15 # Paciente sedado
        else:
            timestep_gcs['gcs'] = timestep_gcs['motor'] + timestep_gcs['verbal'] + timestep_gcs['eyes']
        timestep_gcs = pd.DataFrame(timestep_gcs, index=[0])
        gcs_values_hourly = pd.concat([gcs_values_hourly, timestep_gcs], ignore_index=True)
        timestep += timedelta(hours=1)
    if debug:
        gcsverbal_events.to_csv('./debug/1_{}_gcs_verbal_events.csv'.format(hadm_id))
        gcseyes_events.to_csv('./debug/1_{}_gcs_eyes_events.csv'.format(hadm_id))
        gcsmotor_events.to_csv('./debug/1_{}_gcs_motor_events.csv'.format(hadm_id))
        gcs_values_hourly.to_csv('./debug/1_{}_gcs_hourly.csv'.format(hadm_id))
    return gcs_values_hourly

def get_labs_events(labevents, admittime, dischtime, debug=False, hadm_id=0):
    bilirubin_ids = [50885] # 50885 Bilirubin
    creatinine_ids = [50912] # 50912 Creatinine
    platelet_ids = [51265] # 51265 Platelet

    bilirubin_events = labevents[(labevents['ITEMID'].isin(bilirubin_ids)) & (labevents['VALUENUM'].notna())
                                 & (labevents['VALUENUM'] > 0)]
    bilirubin_events.loc[:, 'VALUENUM'] = np.where(bilirubin_events['VALUENUM'] > 150, np.nan,
                                                   bilirubin_events['VALUENUM'])
    bilirubin_events.loc[:, 'CHARTTIME'] = pd.to_datetime(bilirubin_events['CHARTTIME'], format=datetime_pattern)
    bilirubin_events = bilirubin_events[bilirubin_events['VALUENUM'].notna()]

    creatinine_events = labevents[(labevents['ITEMID'].isin(creatinine_ids)) & (labevents['VALUENUM'].notna())
                                 & (labevents['VALUENUM'] > 0)]
    creatinine_events.loc[:, 'VALUENUM'] = np.where(creatinine_events['VALUENUM'] > 150, np.nan,
                                                    creatinine_events['VALUENUM'])
    creatinine_events.loc[:, 'CHARTTIME'] = pd.to_datetime(creatinine_events['CHARTTIME'], format=datetime_pattern)
    creatinine_events = creatinine_events[creatinine_events['VALUENUM'].notna()]

    platelet_events = labevents[(labevents['ITEMID'].isin(platelet_ids)) & (labevents['VALUENUM'].notna())
                                 & (labevents['VALUENUM'] > 0)]
    platelet_events.loc[:, 'VALUENUM'] = np.where(platelet_events['VALUENUM'] > 10000, np.nan,
                                                    platelet_events['VALUENUM'])
    platelet_events.loc[:, 'CHARTTIME'] = pd.to_datetime(platelet_events['CHARTTIME'], format=datetime_pattern)
    platelet_events = platelet_events[platelet_events['VALUENUM'].notna()]

    timestep = admittime
    labs_values_hourly = pd.DataFrame([])
    while timestep <= dischtime:
        labs_timestep = dict()

        closest_bilirubin_event = []
        if len(bilirubin_events) != 0:
            closest_bilirubin_event = bilirubin_events[bilirubin_events['CHARTTIME'] ==
                                                       min(bilirubin_events['CHARTTIME'],
                                                           key=lambda x: abs(x - timestep))]
        closest_creatinine_event = []
        if len(creatinine_events) !=0:
            closest_creatinine_event = creatinine_events[creatinine_events['CHARTTIME'] ==
                                                       min(creatinine_events['CHARTTIME'],
                                                           key=lambda x: abs(x - timestep))]
        closest_platelet_event = []
        if len(platelet_events) != 0:
            closest_platelet_event = platelet_events[platelet_events['CHARTTIME'] ==
                                                         min(platelet_events['CHARTTIME'],
                                                             key=lambda x: abs(x - timestep))]
        labs_timestep['timestep'] = timestep
        labs_timestep['platelet'] = min(closest_platelet_event['VALUENUM']) if len(closest_platelet_event) != 0 else None
        labs_timestep['creatinine'] = max(closest_creatinine_event['VALUENUM']) if len(closest_creatinine_event) != 0 else None
        labs_timestep['bilirubin'] = max(closest_bilirubin_event['VALUENUM']) if len(closest_bilirubin_event) != 0 else None
        labs_timestep = pd.DataFrame(labs_timestep, index=[0])
        labs_values_hourly = pd.concat([labs_timestep, labs_values_hourly], ignore_index=True)
        timestep += timedelta(hours=1)
    if debug:
        bilirubin_events.to_csv('./debug/2_{}_bilirubin_events.csv'.format(hadm_id))
        creatinine_events.to_csv('./debug/2_{}_creatinine_events.csv'.format(hadm_id))
        platelet_events.to_csv('./debug/2_{}_platelet_events.csv'.format(hadm_id))
        labs_values_hourly.to_csv('./debug/2_{}_labs_hourly.csv'.format(hadm_id))
    return labs_values_hourly

def get_urineoutput_events(outputevents, admittime, dischtime, debug=False, hadm_id=0):
    urine_ids = [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086, 40096, 40651,
                    226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
    urine_events = outputevents[(outputevents['ITEMID'].isin(urine_ids)) & (outputevents['VALUE'].notna())]
    urine_events.loc[:, 'VALUE'] = np.where( np.logical_and(urine_events['ITEMID'] == 227488, urine_events['VALUE'] > 0),
                                             -1*urine_events['VALUE'],
                                             urine_events['VALUE'])
    urine_events.loc[:, 'CHARTTIME'] = pd.to_datetime(urine_events['CHARTTIME'], format=datetime_pattern)

    urineoutput_values_hourly = pd.DataFrame([])
    timestep = admittime
    while timestep <= dischtime:
        urine_timestep = dict()

        closest_urine_output_event = []
        if len(urine_events) != 0:
            closest_urine_output_event =  urine_events[urine_events['CHARTTIME'] ==
                                                       min(urine_events['CHARTTIME'],
                                                           key=lambda x: abs(x - timestep))]

        urine_timestep['timestep'] = timestep
        urine_timestep['urineoutput'] = sum(closest_urine_output_event['VALUE']) if len(closest_urine_output_event) != 0 \
                                            else None
        urine_timestep = pd.DataFrame(urine_timestep, index=[0])
        urineoutput_values_hourly = pd.concat([urineoutput_values_hourly, urine_timestep], ignore_index=True)
        timestep += timedelta(hours=1)
    if debug:
        urine_events.to_csv('./debug/3_{}_urine_events.csv'.format(hadm_id))
        urineoutput_values_hourly.to_csv('./debug/3_{}_urine_hourly.csv'.format(hadm_id))
    return urineoutput_values_hourly

def get_vitals_events(chartevents, admittime, dischtime, debug=False, hadm_id=0):
    mean_arterial_ids =  [ 456, 52, 6702, 443, 220052, 220181, 225312]
    mean_arterial_events = chartevents[(chartevents['ITEMID'].isin(mean_arterial_ids)) & (chartevents['ERROR'] != 1)]
    mean_arterial_events = mean_arterial_events[(mean_arterial_events['VALUENUM'] > 0)
                                                & (mean_arterial_events['VALUENUM'] < 300)]
    mean_arterial_events.loc[:, 'CHARTTIME'] = pd.to_datetime(mean_arterial_events['CHARTTIME'], format=datetime_pattern)

    vitals_hourly_events = pd.DataFrame([])
    timestep = admittime
    while timestep <= dischtime:
        vitals_timestep = dict()
        closest_vitals_event = []
        if len(mean_arterial_events) != 0:
            closest_vitals_event = mean_arterial_events[mean_arterial_events['CHARTTIME'] ==
                                                           min(mean_arterial_events['CHARTTIME'],
                                                               key=lambda x: abs(x - timestep))]
        vitals_timestep['timestep'] = timestep
        vitals_timestep['meanbp'] = min(closest_vitals_event['VALUENUM']) if len(closest_vitals_event) != 0 else None
        vitals_timestep = pd.DataFrame(vitals_timestep, index=[0])
        vitals_hourly_events = pd.concat([vitals_hourly_events, vitals_timestep], ignore_index=True)
        timestep += timedelta(hours=1)
    if debug:
        mean_arterial_events.to_csv('./debug/4_{}_arterial_events.csv'.format(hadm_id))
        vitals_hourly_events.to_csv('./debug/4_{}_vitals_hourly.csv'.format(hadm_id))
    return vitals_hourly_events

def get_respiration_events(chartevents, labevents, admittime, dischtime, debug=False, hadm_id=0):
    so2_ids = [50817]
    fio2_labs_ids = [50816]
    fio2_chart_ids = [3420, 190, 223835, 3422]
    po2_ids = [50821]

    so2_events = labevents[(labevents['ITEMID'].isin(so2_ids)) & (labevents['VALUENUM'] <= 100)]
    so2_events.loc[:, 'CHARTTIME'] = pd.to_datetime(so2_events['CHARTTIME'], format=datetime_pattern)

    po2_events = labevents[(labevents['ITEMID'].isin(po2_ids)) & (labevents['VALUENUM'] <= 800)]
    po2_events.loc[:, 'CHARTTIME'] = pd.to_datetime(po2_events['CHARTTIME'], format=datetime_pattern)

    fio2_lab_events = labevents[labevents['ITEMID'].isin(fio2_labs_ids)]
    fio2_lab_events = fio2_lab_events[(fio2_lab_events['VALUENUM'] >= 20) & (fio2_lab_events['VALUENUM'] <= 100)]
    # fio2_lab_events = fio2_lab_events[]
    fio2_lab_events.loc[:, 'CHARTTIME'] = pd.to_datetime(fio2_lab_events['CHARTTIME'], format=datetime_pattern)

    fio2_chart_events = chartevents[(chartevents['ITEMID'].isin(fio2_chart_ids)) & (chartevents['ERROR'] != 1)]
    fio2_chart_events.loc[:, 'VALUENUM'] = np.where(np.logical_and(fio2_chart_events['ITEMID'] == 223835,
                                                                   np.logical_and(fio2_chart_events['VALUENUM'] > 0,
                                                                                  fio2_chart_events['VALUENUM'] <= 1)),
                                                    fio2_chart_events['VALUENUM']*100, fio2_chart_events['VALUENUM'])
    fio2_chart_events.loc[:, 'VALUENUM'] = np.where(np.logical_and(fio2_chart_events['ITEMID'] == 223835,
                                                                   np.logical_and(fio2_chart_events['VALUENUM'] > 1,
                                                                                  fio2_chart_events['VALUENUM'] < 21)),
                                                    np.nan, fio2_chart_events['VALUENUM'])
    fio2_chart_events.loc[:, 'VALUENUM'] = np.where(np.logical_and(fio2_chart_events['ITEMID'] == 190,
                                                                   np.logical_and(fio2_chart_events['VALUENUM'] > 0.20,
                                                                                  fio2_chart_events['VALUENUM'] <= 1)),
                                                    fio2_chart_events['VALUENUM'] * 100, fio2_chart_events['VALUENUM'])
    fio2_chart_events = fio2_chart_events[fio2_chart_events['VALUENUM'].notna()]
    fio2_chart_events.loc[:, 'CHARTTIME'] = pd.to_datetime(fio2_chart_events['CHARTTIME'], format=datetime_pattern)

    timestep = admittime
    bloodgas_hourly_events = pd.DataFrame([])
    while timestep <= dischtime:
        bloodgas_timestep = dict()
        closest_fio2_lab_event = []
        if len(fio2_lab_events) != 0:
            closest_fio2_lab_event = fio2_lab_events[fio2_lab_events['CHARTTIME'] ==
                                                           min(fio2_lab_events['CHARTTIME'],
                                                               key=lambda x: abs(x - timestep))]

        closest_fio2_chart_event = []
        if len(fio2_chart_events) != 0:
            closest_fio2_chart_event = fio2_chart_events[fio2_chart_events['CHARTTIME'] ==
                                                           min(fio2_chart_events['CHARTTIME'],
                                                               key=lambda x: abs(x - timestep))]
        # Defining which event occurs near to this timestep, the labevents or the chartevents
        closest_fio2_event = []
        try:
            if len(closest_fio2_lab_event) != 0 and len(closest_fio2_chart_event) != 0:
                if closest_fio2_lab_event.loc[closest_fio2_lab_event.index[0], 'CHARTTIME'] \
                        > closest_fio2_chart_event.loc[closest_fio2_chart_event.index[0], 'CHARTTIME']:
                    closest_fio2_event = closest_fio2_lab_event
                else:
                    closest_fio2_event = closest_fio2_chart_event
            elif len(closest_fio2_lab_event) != 0:
                closest_fio2_event = closest_fio2_lab_event
            elif len(closest_fio2_chart_event) != 0:
                closest_fio2_event = closest_fio2_chart_event
        except Exception as e:
            print(e)
            print(closest_fio2_chart_event.index)
            print(closest_fio2_lab_event.index)
            exit()

        closest_po2_event = []
        if len(po2_events) != 0:
            closest_po2_event = po2_events[po2_events['CHARTTIME'] == min(po2_events['CHARTTIME'],
                                                                          key=lambda x: abs(x - timestep))]
        closest_so2_event = []
        if len(so2_events) != 0 :
            closest_so2_event = so2_events[so2_events['CHARTTIME'] == min(so2_events['CHARTTIME'],
                                                                          key=lambda x: abs(x - timestep))]

        bloodgas_timestep['timestep'] = timestep
        bloodgas_timestep['fio2'] = max(closest_fio2_event['VALUENUM']) if len(closest_fio2_event) != 0 else 0
        bloodgas_timestep['po2'] = max(closest_po2_event['VALUENUM']) if len(closest_po2_event) != 0 else 0
        bloodgas_timestep['so2'] = max(closest_so2_event['VALUENUM']) if len(closest_so2_event) != 0 else 0
        if bloodgas_timestep['fio2'] == 0:
            bloodgas_timestep['sao2fio2'] = None
            bloodgas_timestep['pao2fio2'] = None
        else:
            bloodgas_timestep['sao2fio2'] = 100*bloodgas_timestep['so2']/bloodgas_timestep['fio2']
            bloodgas_timestep['pao2fio2'] = 100*bloodgas_timestep['po2']/bloodgas_timestep['fio2']
        bloodgas_timestep = pd.DataFrame(bloodgas_timestep, index=[0])
        bloodgas_hourly_events = pd.concat([bloodgas_hourly_events, bloodgas_timestep], ignore_index=True)
        timestep += timedelta(hours=1)
    if debug:
        fio2_chart_events.to_csv('./debug/5_{}_fio2_chart_events.csv'.format(hadm_id))
        fio2_lab_events.to_csv('./debug/5_{}_fio2_lab_events.csv'.format(hadm_id))
        po2_events.to_csv('./debug/5_{}_po2_events.csv'.format(hadm_id))
        so2_events.to_csv('./debug/5_{}_so2_events.csv'.format(hadm_id))
        bloodgas_hourly_events.to_csv('./debug/5_{}_bloodgas_hourly.csv'.format(hadm_id))
    return bloodgas_hourly_events

def get_vasopressor_events(inputevents, weights, echodata, admittime, dischtime, debug=False, hadm_id=0):
    vasopressor_ids = [30047, 30120, 30044, 30119, 30309, 30043, 30307, 30042, 30306, 221906, 221289, 221662, 221653]
    norepinephrine_ids = [30047, 30120, 221906]
    epinephrine_ids = [30044, 30119, 30309, 221289]
    dopamine_ids = [30043, 30307, 221662]
    dobutamine_ids = [30042, 30306, 221653]
    divide_rate_weight_ids = [30047, 30044]
    admit_vasopressor_events = inputevents[(inputevents['ITEMID'].isin(vasopressor_ids)) & (inputevents['RATE'].notna())]
    # Transform datetime
    admit_vasopressor_events.loc[:, 'CHARTTIME'] = pd.to_datetime(admit_vasopressor_events['CHARTTIME'],
                                                           format=datetime_pattern)
    echodata.loc[:, 'charttime'] = pd.to_datetime(echodata['charttime'], format=datetime_pattern)
    # Get patient weight for each vasopressor charttime
    aux_weights = []
    for index, vaso_event in admit_vasopressor_events.iterrows():

        if weights is not None and len(weights['CHARTTIME']) != 0:
            weight = weights[weights['CHARTTIME'] == min(weights['CHARTTIME'],
                                                         key=lambda x: abs(x - vaso_event['CHARTTIME']))]
            weight = sum(weight['VALUENUM'])/len(weight)
        else:
            if len(echodata) != 0:
                try:
                    weight = echodata[echodata['charttime'] == min(echodata['charttime'],
                                                                 key=lambda x: abs(x - vaso_event['CHARTTIME']))]
                    weight = sum(weight['weight']) / len(weight)
                except:
                    print(type(echodata.loc[0, 'charttime']))
                    exit()
            else:
                weight = np.nan
        aux_weights.append(weight)
    # Transform for the ids that are not measured by the patient weight
    admit_vasopressor_events.loc[:, 'weight'] = aux_weights
    # Removing events that need to be divided by weight but weight is equal no nan
    admit_vasopressor_events = admit_vasopressor_events[
        ~(admit_vasopressor_events['ITEMID'].isin(divide_rate_weight_ids))
        | (admit_vasopressor_events['weight'].notna())]
    admit_vasopressor_events['RATE'] = np.where(admit_vasopressor_events['ITEMID'].isin(divide_rate_weight_ids),
                                                admit_vasopressor_events['RATE'] / admit_vasopressor_events[
                                                    'weight'],
                                                admit_vasopressor_events['RATE'])
    admit_norepinephrine_rate = \
                    admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(norepinephrine_ids)][
                        ['CHARTTIME', 'RATE']]
    admit_norepinephrine_rate['CHARTTIME'] = pd.to_datetime(admit_norepinephrine_rate['CHARTTIME'],
                                                            format=datetime_pattern)

    admit_epinephrine_rate = admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(epinephrine_ids)][
        ['CHARTTIME', 'RATE']]
    admit_epinephrine_rate['CHARTTIME'] = pd.to_datetime(admit_epinephrine_rate['CHARTTIME'],
                                                         format=datetime_pattern)

    admit_dopamine_rate = admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(dopamine_ids)][
        ['CHARTTIME', 'RATE']]
    admit_dopamine_rate['CHARTTIME'] = pd.to_datetime(admit_dopamine_rate['CHARTTIME'], format=datetime_pattern)

    admit_dobutamine_rate = admit_vasopressor_events[admit_vasopressor_events['ITEMID'].isin(dobutamine_ids)][
        ['CHARTTIME', 'RATE']]
    admit_dobutamine_rate['CHARTTIME'] = pd.to_datetime(admit_dobutamine_rate['CHARTTIME'], format=datetime_pattern)

    timestep = admittime
    vasopressor_hourly_events = pd.DataFrame([])
    while timestep <= dischtime:
        closest_epinephrine_event = []
        if len(admit_epinephrine_rate) != 0:
            closest_epinephrine_event = admit_epinephrine_rate[admit_epinephrine_rate['CHARTTIME'] ==
                                                               min(admit_epinephrine_rate['CHARTTIME'],
                                                                   key=lambda x: abs(x - timestep))]
        closest_norepinephrine_event = []
        if len(admit_norepinephrine_rate) != 0:
            closest_norepinephrine_event = admit_norepinephrine_rate[admit_norepinephrine_rate['CHARTTIME'] ==
                                                               min(admit_norepinephrine_rate['CHARTTIME'],
                                                                   key=lambda x: abs(x - timestep))]
        closest_dobutamine_event = []
        if len(admit_dobutamine_rate) != 0:
            closest_dobutamine_event = admit_dobutamine_rate[admit_dobutamine_rate['CHARTTIME'] ==
                                                               min(admit_dobutamine_rate['CHARTTIME'],
                                                                   key=lambda x: abs(x - timestep))]
        closest_dopamine_event = []
        if len(admit_dopamine_rate) != 0:
            closest_dopamine_event = admit_dopamine_rate[admit_dopamine_rate['CHARTTIME'] ==
                                                               min(admit_dopamine_rate['CHARTTIME'],
                                                                   key=lambda x: abs(x - timestep))]
        vasopressor_timestep = dict()
        vasopressor_timestep['timestep'] = timestep
        vasopressor_timestep['epinephrine'] = max(closest_epinephrine_event['RATE']) if len(closest_epinephrine_event) \
                                                                                        != 0 else None
        vasopressor_timestep['norepinephrine'] = max(closest_norepinephrine_event['RATE']) if len(closest_norepinephrine_event) \
                                                                                        != 0 else None
        vasopressor_timestep['dobutamine'] = max(closest_dobutamine_event['RATE']) if len(closest_dobutamine_event) \
                                                                                        != 0 else None
        vasopressor_timestep['dopamine'] = max(closest_dopamine_event['RATE']) if len(closest_dopamine_event) \
                                                                                        != 0 else None
        vasopressor_timestep = pd.DataFrame(vasopressor_timestep, index=[0])
        vasopressor_hourly_events = pd.concat([vasopressor_hourly_events, vasopressor_timestep], ignore_index=True)
        timestep += timedelta(hours=1)
    if debug:
        admit_norepinephrine_rate.to_csv('./debug/6_{}_norepinephrine_events.csv'.format(hadm_id))
        admit_epinephrine_rate.to_csv('./debug/6_{}_epinephrine_events.csv'.format(hadm_id))
        admit_dopamine_rate.to_csv('./debug/6_{}_dopamine_events.csv'.format(hadm_id))
        admit_dobutamine_rate.to_csv('./debug/6_{}_dobutamine_events.csv'.format(hadm_id))
        vasopressor_hourly_events.to_csv('./debug/6_{}_vasopressor_hourly.csv'.format(hadm_id))
    return vasopressor_hourly_events