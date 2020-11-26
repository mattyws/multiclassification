import os
parameters = {
    # "mimic_data_path" : os.path.expanduser("~/Documents/mimic/"),
    "mimic_data_path" : "/scratch/mfgrawe/mimic",
    "csv_files_directory" : "csv/",
    "multiclassification_directory" : "multiclassification/",
    "all_stays_csv" : "all_stays.csv",
    "all_stays_csv_w_events" : "all_stays_w_events.csv",

    "decompensation_directory" : "decompensation/",
    "decompensation_dataset_csv" : "decompensation.csv",

    "mortality_directory" : "mortality/",
    "mortality_dataset_csv" : "mortality.csv",

    "raw_events_dirname" : "{}_timeseries/",
    "merged_events_dirname": "merged_timeseries/",
    "features_filtered_dirname": "filtered_timeseries/",
    "events_hourly_merged_dirname" : "bucket_timeseries/",
    "noteevents_anonymized_tokens_normalized" : "textual_anonymized_data/",
    "noteevents_anonymized_tokens_normalized_preprocessed" : "textual_normalized_preprocessed/",
    "noteevents_hourly_merged_dirname" : "textual_hourly_merged/",

    "structured_dirname" : "structured_data/",
    "textual_dirname" : "textual_data/",
    
    "hotencoded_events_dirname" : "hotencoded_undersampled_{}/",
    "all_features_dirname" : "all_features_undersampled_{}/",
    "separated_features_types_dirname": "separated_features/",
    "features_low_frequency_removed_dirname": "low_frequency_removed/",
    "patients_features_dirname" : "{}_features_undersampled/",
    "tokenized_noteevents_dirname" : "tokenized_noteevents/",
    "ctakes_data_path" : "noteevents_ctakes/",
    "ctakes_output_path" : "noteevents_ctakes_output/",
    "noteevents_ctakes_processed_data_path" : "noteevents_processed_ctakes/",
    "noteevents_cuis_normalized_sentences" : "noteevents_sentences/",
    "bag_of_cuis_files_path" : "bag_of_cuis_files_path/",

    "features_after_binarized_file_name" : "features_after_binarized_undersampled.pkl",
    "features_types_file_name" : "features_types.pkl",
    "values_for_mixed_in_patient_file_name" : "mixed_features_values.pkl",
    "features_frequency_file_name" : "features_frequency.pkl",

    "date_pattern" : "%Y-%m-%d",
    "datetime_pattern" : "%Y-%m-%d %H:%M:%S",

    "notes_tokens_features_file_name": "notes_tokens_features.pkl",
    
    'ctakes_input_dir': 'ctakes_input_dir/',
    'ctakes_output_path': 'ctakes_output_dir/',
    'ctakes_processed_data_path': 'textual_ctakes/'
}