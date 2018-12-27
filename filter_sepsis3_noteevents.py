import os
import json
import functions
import pprint

pp = pprint.PrettyPrinter(indent=5)

sepsis3_json_files_paths  = ["/home/mattyws/Documents/json_sepsis", "/home/mattyws/Documents/json_no_sepsis"]

max_hours_since_admission = 12

total_files = 0
for sepsis3_json_files_path in sepsis3_json_files_paths:
    for dir, path, files in os.walk(sepsis3_json_files_path):
        if "no_sepsis" in dir:
            dir_noteevents = dir.replace("json_no_sepsis", "no_sepsis_noteevents")
        else:
            dir_noteevents = dir.replace("json_sepsis", "sepsis_noteevents")
        if not os.path.exists(dir_noteevents):
            os.mkdir(dir_noteevents)
        for file in files:
            total_files += 1
            if total_files % 100 == 0:
                print("{} files processed.".format(total_files))
            with open(dir+'/'+file) as json_file_handler:
                try:
                    admission = json.load(json_file_handler)
                    if "noteevents" in admission.keys():
                        if "no_sepsis" in dir:
                            time_to_use = admission["admittime"]
                            filtered_noteevents = functions.filter_since_time(admission['noteevents'], time_to_use,
                                                                              max_hours_since_admission, key="charttime", after=True)
                        else:
                            time_to_use = admission["sepsis3"]["suspected_infection_time_poe"]
                            filtered_noteevents = functions.filter_since_time(admission['noteevents'], time_to_use,
                                                                              max_hours_since_admission, key="charttime")
                        if len(filtered_noteevents) > 0:
                            with open(dir_noteevents+"/"+file, 'w') as noteevents_file_handler:
                                json.dump(filtered_noteevents, noteevents_file_handler)
                except:
                    print("{} failed to process".format(file))