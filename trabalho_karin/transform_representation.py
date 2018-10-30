import json
import time




date_pattern = "%Y-%m-%d"
datetime_pattern = "%Y-%m-%d %H:%M:%S"

def get_data_from_admitday(json_object, date_str, key="charttime", date=False):
    admittime = time.strptime(date_str, datetime_pattern)
    filtered_objects = []
    for event in json_object:
        if date:
            event_date = time.strptime(event[key], date_pattern)
        else:
            event_date = time.strptime(event[key], datetime_pattern)
        difference = time.mktime(event_date) - time.mktime(admittime)
        if abs(difference) / 86400 <= 1:
            filtered_objects.append(event)
        # break
    return filtered_objects

with open('sepsis_patients4', 'r') as patients_w_sepsis_handler:
    for line in patients_w_sepsis_handler:
        patient = patients_w_sepsis_handler.readline()
        patient = json.load(open(line.strip(), 'r'))
        print(patient['admittime'])
        filtered_object = get_data_from_admitday(patient['prescriptions'], patient['admittime'], key='startdate', date=False)
        break