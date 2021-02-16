import pandas as pd
import pandas
import numpy as np
from scipy.stats import zscore
import os

from sklearn.model_selection import StratifiedKFold

from multiclassification.classification_ensemble_stacking import train_meta_model_on_data, test_meta_model_on_data
from multiclassification.parameters.classification_parameters import ensemble_parameters as parameters
import pickle
from functions import remove_empty_textual_data_episodes, print_with_time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
problem = 'mortality'
training_base_directory = parameters['multiclassification_base_path'] + parameters['training_directory_path']
training_directory = training_base_directory + parameters[problem+"_directory"] \
                     + "metadata_test/"
checkpoint_directory = training_directory + parameters['training_checkpoint']

if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)

with open(checkpoint_directory + "parameters.pkl", 'wb') as handler:
    pickle.dump(parameters, handler)

# Loading csv
print_with_time("Loading data")
dataset_path = os.path.join(parameters['multiclassification_base_path'], parameters[problem+'_directory'],
                            parameters[problem+'_dataset_csv'])

data_csv = pd.read_csv(dataset_path)
data_csv = data_csv.sort_values(['episode'])
data_csv = remove_empty_textual_data_episodes(data_csv, 'textual_path')
print(data_csv['label'].value_counts())
metadata_columns = ['age', 'is_male', 'height', 'weight']
dataset_patients = pandas.read_csv('/home/mattyws/Documents/mimic/sepsis3-df-no-exclusions.csv')
dataset_patients[['age', 'is_male', 'height', 'weight']] = dataset_patients[['age', 'is_male', 'height', 'weight']].apply(zscore)
for column in metadata_columns:
    print("======================================= {} ============================================".format(column))
    print(dataset_patients[column])
    len_total = len(dataset_patients[column])
    len_missing = len(dataset_patients[dataset_patients[column].isna()])
    print("{} with total of {} values and missing {}".format(column, len_total, len_missing))
exit()
dataset_patients[['age', 'is_male', 'height', 'weight']] = dataset_patients[['age', 'is_male', 'height', 'weight']].fillna(0)
meta_data_extra = dataset_patients[['icustay_id', 'age', 'is_male', 'height', 'weight']]

# Testing only structured non-temporal data
metadata_dataset = pandas.merge(meta_data_extra, data_csv[['episode', 'icustay_id', 'label']], left_on="icustay_id", right_on="icustay_id")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
classes = np.asarray(metadata_dataset['label'].tolist())
data = np.asarray(metadata_dataset["episode"].tolist())
fold = 0
results_df = None
for trainIndex, testIndex in kf.split(data, classes):
    training_data = metadata_dataset[metadata_dataset['episode'].isin(data[trainIndex])]
    training_classes = training_data['label'].tolist()
    training_data = training_data[['age', 'is_male', 'height', 'weight']].values
    test_data = metadata_dataset[metadata_dataset['episode'].isin(data[testIndex])]

    meta_adapters = train_meta_model_on_data(training_data, training_classes, parameters)
    results, result_predictions = test_meta_model_on_data(meta_adapters, test_data,
                                                          ['age', 'is_male', 'height', 'weight'], 'metadata')
    result_predictions = pandas.DataFrame(result_predictions)
    result_predictions.to_csv(os.path.join(checkpoint_directory, "metamodel_metadata_predictions.csv"))
    results = pandas.DataFrame(results)
    results['fold'] = fold
    if results_df is None:
        results_df = results_df
    else:
        results_df = pd.concat([results_df, results])
    results.to_csv(os.path.join(checkpoint_directory, 'results_{}.csv'.format(fold)))
    fold += 1

results_df.to_csv(os.path.join(checkpoint_directory, 'metadata_folds_results.csv'))
results_groupby = results_df.groupby(by=["model"])
results_groupby.mean().to_csv(os.path.join(checkpoint_directory, 'metadata_mean_results.csv'))