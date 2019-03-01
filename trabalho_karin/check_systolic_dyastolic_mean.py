import json
import time
import csv
import trabalho_karin.helper as helper

microbiologyevent_label = "microbiologyevents"
chartevents_label = 'chartevents'

charttime_label = 'charttime'
itemid_label = 'ITEMID'

patient_file = 'PATIENTS.csv'
datetime_pattern = "%Y-%m-%d %H:%M:%S"
birth_label = 'DOB'

def get_patient_age(patient_id, admittime_str):
    admittime = time.strptime(admittime_str, datetime_pattern)
    with open(patient_file, 'r') as patient_file_handler:
        dict_reader = csv.DictReader(patient_file_handler)
        for row in dict_reader:
            if row['subject_id'.upper()] == patient_id:
                dob = time.strptime(row[birth_label], datetime_pattern)
                difference = admittime.tm_year - dob.tm_year - ((admittime.tm_mon, dob.tm_mday) < (admittime.tm_mon, dob.tm_mday))
                return difference
    return None

'''
Para cada paciente, adquire todos os ids que foram medidos para cada visita de um profissional de saúde,
e verifica a existência dos ids para pressão sistólica e diastólica, caso possua, verifica a existência dos
ids de média de pressão
'''
diastolic_set = set(helper.DIASTOLIC_IDS)
systolic_set = set(helper.SYSTOLIC_IDS)
mean_set = set(helper.PRESSURE_IDS)


with open('sepsis_patients4', 'r') as patients_w_sepsis_handler:
    for line in patients_w_sepsis_handler:
        print(line.strip().split('/')[-1])
        patient = json.load(open(line.strip(), 'r'))
        patient_age = get_patient_age(patient['subject_id'], patient['admittime'])
        events_in_patient = dict()
        if microbiologyevent_label in patient.keys() and (patient_age > 18 and patient_age < 80):
            for events in patient[chartevents_label]:
                if events[charttime_label] not in events_in_patient.keys():
                    events_in_patient[events[charttime_label]] = set()
                events_in_patient[events[charttime_label]].add(events[itemid_label])

        for time_key in events_in_patient.keys():
            has_systolic = len(events_in_patient[time_key].intersection(systolic_set)) > 0
            has_diastolic = len(events_in_patient[time_key].intersection(diastolic_set)) > 0
            has_mean = len(events_in_patient[time_key].intersection(mean_set)) > 0
            if has_systolic and has_diastolic and not has_mean :
                print("Não tem mean ----------------------------------------------")