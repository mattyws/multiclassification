import os
import shutil

from resources import functions

parameters = functions.load_parameters_file()

ctakes_input_dir = parameters['mimic_data_path'] + "sepsis_noteevents_ctakes/"
ctakes_tmp_dir = parameters['mimic_data_path'] + "sepsis_noteevents_ctakes_tmp/"
if not os.path.exists(ctakes_tmp_dir):
    os.mkdir(ctakes_tmp_dir)
ctakes_output_dir = parameters['mimic_data_path'] + "sepsis_noteevents_ctakes_output/"

listdir_input = set(os.listdir(ctakes_input_dir))
listdir_output = set(os.listdir(ctakes_output_dir))

not_processed = listdir_input.difference(listdir_output)
processed = list(listdir_input.intersection(listdir_output))

# Check if all files inside processed directories were processed, if not, add the directory to the not processed
for directory in processed:
    len_input = len(os.listdir(ctakes_input_dir + directory))
    len_output = len(os.listdir(ctakes_output_dir + directory))
    if len_input != len_output:
        not_processed.add(directory)
    else:
        print(directory)
        # Move the processed directory to temporary directory
        shutil.move(ctakes_input_dir + directory, ctakes_tmp_dir + directory)