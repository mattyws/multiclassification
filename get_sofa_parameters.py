from datetime import timedelta

import numpy as np
import pandas as pd

datetime_pattern = "%Y-%m-%d %H:%M:%S"

# def get_pao2fio2_events(labevents, chartevents)


def get_gcs_events(chartevents, admittime, dischtime):
    GCS_ids = [184, 454, 723, 223900, 223901, 220739]
    GCSVerbal_ids = [723, 223900]
    GCSMotor_ids = [454, 223901]
    GCSEyes_ids = [184, 220739]

    gcsverbal_events = chartevents[chartevents['ITEMID'].isin(GCSVerbal_ids)]
    # These values indicates that the patient is in endotrach/vent
    gcsverbal_events.loc[:, 'VALUENUM'] = np.where(gcsverbal_events['VALUE'] == '1.0 ET/Trach', 0,
                                               gcsverbal_events['VALUENUM'])
    gcsverbal_events.loc[:, 'VALUENUM'] = np.where(gcsverbal_events['VALUE'] == 'No Response-ETT', 0,
                                               gcsverbal_events['VALUENUM'])
    gcsverbal_events.loc[:, 'CHARTTIME'] = pd.to_datetime(gcsverbal_events['CHARTTIME'], format=datetime_pattern)

    gcsmotor_events = chartevents[chartevents['ITEMID'].isin(GCSMotor_ids)]
    gcsmotor_events.loc[:, 'CHARTTIME'] = pd.to_datetime(gcsmotor_events['CHARTTIME'], format=datetime_pattern)

    gcseyes_events = chartevents[chartevents['ITEMID'].isin(GCSEyes_ids)]
    gcseyes_events.loc[:, 'CHARTTIME'] = pd.to_datetime(gcseyes_events['CHARTTIME'], format=datetime_pattern)

    timestep = admittime
    gcs_values_hourly = pd.DataFrame([])
    print(gcsverbal_events)
    print(gcsmotor_events)
    print(gcseyes_events)
    while timestep <= dischtime:
        timestep_gcs = dict()
        timestep_gcs['timestep'] = timestep

        closest_verbal_event = gcsverbal_events[gcsverbal_events['CHARTTIME'] ==
                                                min(gcsverbal_events['CHARTTIME'], key=lambda x: abs(x - timestep))].iloc[0]
        closest_motor_event = gcsmotor_events[gcsmotor_events['CHARTTIME'] ==
                                             min(gcsmotor_events['CHARTTIME'], key=lambda x: abs(x - timestep))].iloc[0]
        closest_eyes_event = gcseyes_events[gcseyes_events['CHARTTIME'] ==
                                            min(gcseyes_events['CHARTTIME'], key=lambda x: abs(x - timestep))].iloc[0]

        timestep_gcs['motor'] = closest_motor_event['VALUENUM']
        timestep_gcs['verbal'] = closest_verbal_event['VALUENUM']
        timestep_gcs['eyes'] = closest_eyes_event['VALUENUM']
        if timestep_gcs['verbal'] is None or (timestep_gcs['verbal'] is not None and timestep_gcs['verbal'] == 0):
            timestep_gcs['gcs'] = 15 # Paciente sedado
        else:
            timestep_gcs['gcs'] = timestep_gcs['motor'] + timestep_gcs['verbal'] + timestep_gcs['eyes']

        print(timestep_gcs['timestep'], timestep_gcs['gcs'])
        timestep += timedelta(hours=1)