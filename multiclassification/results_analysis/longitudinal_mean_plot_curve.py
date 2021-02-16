import pandas as pd
from tqdm import tqdm
import os

from resources.functions import print_with_time

def get_mean_for_tableone(dataset:pd.DataFrame, file_path_column:str):
    print("Calculating means")
    values = dict()
    values_positive = dict()
    values_negative = dict()
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        episode_df = pd.read_csv(row[file_path_column])
        episode_df = episode_df.set_index(["bucket"], drop=False)
        if "Unnamed: 0" in episode_df.columns:
            episode_df = episode_df.drop(columns=["Unnamed: 0"])
        for column in episode_df.columns:
            if "Unnamed" in column or column == "starttime" or column == "endtime" or '_min' in column or '_max' in column:
                continue
            if column not in values.keys():
                values[column] = []
            values[column].extend(episode_df[column].dropna().tolist())
            if row['label'] == 1:
                if column not in values_positive.keys():
                    values_positive[column] = []
                values_positive[column].extend(episode_df[column].dropna().tolist())
            if row['label'] == 0:
                if column not in values_negative.keys():
                    values_negative[column] = []
                values_negative[column].extend(episode_df[column].dropna().tolist())
    means = []
    for key in values.keys():
        means.append({"feature":key,
                      "total":sum(values[key])/len(values[key]),
                      "positive":sum(values_positive[key])/len(values_positive[key]),
                      "negative":sum(values_negative[key])/len(values_negative[key])
        })
    means = pd.DataFrame(means)
    return means

def get_means(dataset:pd.DataFrame, file_path_column:str):
    print("Calculating means")
    mean_df = None
    num_values = dict()
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        episode_df = pd.read_csv(row[file_path_column])
        # episode_df = episode_df.set_index(["bucket"], drop=False)
        if "Unnamed: 0" in episode_df.columns:
            episode_df = episode_df.drop(columns=["Unnamed: 0"])
        if mean_df is None:
            mean_df = episode_df
        else:
            for column in episode_df.columns:
                if "Unnamed" in column or column == "starttime" or column == "endtime" \
                        or '_min' in column or '_max' in column or column == "bucket":
                    continue
                if column not in num_values.keys():
                    num_values[column] = [0 for x in range(48)]
                mean_df.loc[:, column] = mean_df[column].add(episode_df[column], fill_value=0)
                for n, value in enumerate(episode_df[column].values):
                    if not pd.isna(value):
                        num_values[column][n] += 1
    raw_df = mean_df.copy()
    for column in mean_df:
        if "Unnamed" in column or column == "starttime" or column == "endtime" \
                or '_min' in column or '_max' in column or column == "bucket":
            continue
        try:
            mean_df.loc[:, column] = mean_df[column].div(num_values[column][:len(mean_df)])
        except Exception as e:
            print(len(mean_df))
            print(len(num_values[column]))
            raise e
    return mean_df, raw_df, num_values

if __name__ == "__main__":
    from multiclassification.parameters.classification_parameters import ensemble_parameters as parameters

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    problem = 'mortality'
    training_base_directory = parameters['multiclassification_base_path'] + parameters['training_directory_path']
    training_directory = training_base_directory + parameters[problem + "_directory"] \
                         + parameters['execution_saving_path']
    checkpoint_directory = training_directory + parameters['training_checkpoint']
    analysis_directory = os.path.join(training_directory, "analysis/")
    if not os.path.exists(analysis_directory):
        os.mkdir(analysis_directory)
    print_with_time("Loading data")
    dataset_path = os.path.join(parameters['multiclassification_base_path'], parameters[problem + '_directory'],
                                parameters[problem + '_dataset_csv'])

    data_csv = pd.read_csv(dataset_path)
    data_csv = data_csv.sort_values(['episode'])
    positive_samples = data_csv[data_csv['label'] == 1]
    get_means(positive_samples, 'structured_path')

