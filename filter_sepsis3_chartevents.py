import json
import os
import pprint
from datetime import datetime, timedelta

import functions
#TODO : filtrar adimissões que a hora da infecção tenha ocorrido com no mínimo Xh após a internação do paciente
pp = pprint.PrettyPrinter(indent=5)
preciding_time_since_infection = 4

sepsis3_json_files_paths  = ["/home/mattyws/Documentos/mimic/data/json_sepsis" ] #, "/home/mattyws/Documents/json_no_sepsis"]
datetime_pattern = "%Y-%m-%d %H:%M:%S"

total_files = 0
total_patient_infection_before_adm = 0
for sepsis3_json_files_path in sepsis3_json_files_paths:
    for dir, path, files in os.walk(sepsis3_json_files_path):
        if "no_sepsis" in dir:
            dir_chartevents = dir.replace("json_no_sepsis", "no_sepsis_chartevents")
        else:
            dir_chartevents = dir.replace("json_sepsis", "sepsis_chartevents")
        if not os.path.exists(dir_chartevents):
            os.mkdir(dir_chartevents)
        for file in files:
            total_files += 1
            if total_files % 100 == 0:
                print("{} files processed.".format(total_files))
            with open(dir + '/' + file) as json_file_handler:
                try:
                    admission = json.load(json_file_handler)
                    if "chartevents" in admission.keys():
                        if "no_sepsis" in dir:
                            time_to_use = admission["admittime"]
                            filtered_chartevents = functions.filter_since_time(admission['chartevents'], time_to_use,
                                                                               12,
                                                                               key="charttime", after=True)
                        else:
                            admittime = admission["admittime"]
                            infection_time = admission["sepsis3"]["suspected_infection_time_poe"]
                            # print(admission['hadm_id'], admittime, infection_time)
                            # filtered_chartevents = functions.filter_events_before_infection(admission['chartevents'], admittime,
                            #                                                                 infection_time, preciding_time_since_infection,
                            #                                                                 time_key="charttime")
                            admittime_datetime = datetime.strptime(admittime, datetime_pattern)
                            infection_datetime = datetime.strptime(infection_time, datetime_pattern) - timedelta(
                                hours=preciding_time_since_infection)
                            if infection_datetime < admittime_datetime :
                                total_patient_infection_before_adm += 1
                        # for event in filtered_chartevents:
                        #     print(event['charttime'])
                        # if len(filtered_noteevents) > 0:
                        #     with open(dir_chartevents + "/" + file, 'w') as noteevents_file_handler:
                        #         json.dump(filtered_noteevents, noteevents_file_handler)
                except Exception as e:
                    print("{} failed to process".format(file), e)
print(total_patient_infection_before_adm)