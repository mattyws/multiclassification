import csv
import json

import pandas

def get_organism_class(events):
    org_item_label = "ORG_ITEMID"
    interpretation_label = "interpretation"
    organism_count = dict()
    for event in events:
        if org_item_label in event.keys():
            if event[org_item_label] not in organism_count.keys():
                organism_count[event[org_item_label]] = 0
            if event[interpretation_label] == 'R':
                organism_count[event[org_item_label]] += 1
                if organism_count[event[org_item_label]] == 3:
                    return "R"
    return "S"

data = pandas.read_csv('sepsis_file2.csv')
data_resistent = data[data['organism_resistence'] == 'R']
data_nonresistent = data[data['organism_resistence'] == 'S']

data_resistent_age = data_resistent[data_resistent["age"] >= 18]
data_resistent_age = data_resistent_age[data_resistent_age["age"] <= 80]

data_nonresistent_age = data_nonresistent[data_nonresistent["age"] >= 18]
data_nonresistent_age = data_nonresistent_age[data_nonresistent_age["age"] <= 80]

print("Quantidade de pacientes com bactérias resistentes", len(data_resistent))
print("Quantidade de pacientes com bactérias não resistentes", len(data_nonresistent))
print("Média de idade")
print('Resistente', data_resistent_age["age"].mean())
print('Não resistente', data_nonresistent_age["age"].mean())
print("Distribuição genero")
print("Resistente", data_resistent['GENDER'].value_counts().to_dict())
print("Não resistente", data_nonresistent['GENDER'].value_counts().to_dict())
# print("Etinicidade")
# print("Resistente", data_resistent['ethnicity'].value_counts().to_dict())
# print("Não resistente", data_nonresistent['ethnicity'].value_counts().to_dict())
print("Uso de vasopressores")
print("Resistente", data_resistent['vasopressor'].value_counts().to_dict())
print("Não resistente", data_nonresistent['vasopressor'].value_counts().to_dict())
print("Média de SOFA")
print("Resistente", data_resistent['sofa'].mean())
print("Não resistente", data_nonresistent['sofa'].mean())
print("Ventilação mecânica")
print("Resistente", data_resistent['item_467'].value_counts().to_dict())
print("Não resistente", data_nonresistent['item_467'].value_counts().to_dict())

dict_patients = dict()
print("Loading patients")
with open('PATIENTS.csv', 'r') as patients_csv_handler:
    dict_reader = csv.DictReader(patients_csv_handler)
    for row in dict_reader:
        dict_patients[row["SUBJECT_ID"]] = row


mortality_resistent = dict()
mortality_nonresistent = dict()
with open('sepsis_patients4', 'r') as patients_w_sepsis_handler:
    for line in patients_w_sepsis_handler:
        print(line.strip().split('/')[-1])
        patient = json.load(open(line.strip(), 'r'))

        if "microbiologyevents" in patient.keys():
            if get_organism_class(patient["microbiologyevents"]) == 'R':
                if dict_patients[patient["subject_id"]]["EXPIRE_FLAG"] not in mortality_resistent.keys():
                    mortality_resistent[dict_patients[patient["subject_id"]]["EXPIRE_FLAG"]] = 0
                mortality_resistent[dict_patients[patient["subject_id"]]["EXPIRE_FLAG"]] += 1
            else:
                if dict_patients[patient["subject_id"]]["EXPIRE_FLAG"] not in mortality_nonresistent.keys():
                    mortality_nonresistent[dict_patients[patient["subject_id"]]["EXPIRE_FLAG"]] = 0
                mortality_nonresistent[dict_patients[patient["subject_id"]]["EXPIRE_FLAG"]] += 1

print("Contagem mortalidade resistente", mortality_resistent)
print("Contagem mortalidade não resistente", mortality_nonresistent)