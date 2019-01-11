import json
import os

root = "/home/mattyws/Documentos/mimic/data/json/"

with open("sepsis_patients4", 'w', buffering=1) as patients_file_handler:
	for path, subdirs, files in os.walk(root):
		for name in files:
			f = os.path.join(path, name)
			with open(f, 'r') as json_file_handler:
				json_object = json.load(json_file_handler)
				if "diagnoses_icd" in json_object.keys():
					for diagnoses in json_object['diagnoses_icd']:
						if diagnoses['seq_num'] == '1' and (diagnoses['icd9_code'] == '99591' or
															diagnoses['icd9_code'] == '99592' or
															diagnoses['icd9_code'].startswith('038')):
							print(f, diagnoses)
							patients_file_handler.write(f + '\n')
							patients_file_handler.flush()	
							break


