import json
import os
import pprint

import functions
#TODO : change script to chartevents
pp = pprint.PrettyPrinter(indent=5)
max_hours_since_admission = 12

sepsis3_json_files_paths  = ["/home/mattyws/Documents/json_sepsis", "/home/mattyws/Documents/json_no_sepsis"]

total_files = 0
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
                            filtered_noteevents = functions.filter_since_time(admission['chartevents'], time_to_use,
                                                                              max_hours_since_admission,
                                                                              key="charttime", after=True)
                        else:
                            time_to_use = admission["sepsis3"]["suspected_infection_time_poe"]
                            filtered_noteevents = functions.filter_since_time(admission['chartevents'], time_to_use,
                                                                              max_hours_since_admission,
                                                                              key="charttime")
                        if len(filtered_noteevents) > 0:
                            with open(dir_chartevents + "/" + file, 'w') as noteevents_file_handler:
                                json.dump(filtered_noteevents, noteevents_file_handler)
                except:
                    print("{} failed to process".format(file))