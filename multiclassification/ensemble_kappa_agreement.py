import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

from functions import remove_empty_textual_data_episodes
from multiclassification.parameters.classification_parameters import ensemble_parameters as parameters
import os


def kappa_aggreement(predictions:pd.DataFrame, dataset:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    for column in all_predictions.columns:
        if 'Unnamed' in column:
            continue
        all_predictions.loc[:, column] = all_predictions[column].apply(lambda x: 1 if x>0.5 else 0)
    data = dataset[['episode', 'label']]
    predictions = pd.merge(predictions, data, left_index=True, right_on='episode')
    kappas = dict()
    kappas_positive = dict()
    kappas_negative = dict()
    for col in predictions.columns:
        if 'Unnamed' in col or col == 'label' or col == 'episode':
            continue
        if col not in kappas.keys():
            kappas[col] = dict()
        if col not in kappas_positive.keys():
            kappas_positive[col] = dict()
        if col not in kappas_negative.keys():
            kappas_negative[col] = dict()
        for col2 in predictions.columns:
            if 'Unnamed' in col2 or col2 == 'label' or col == 'episode':
                continue
            y_col = predictions.loc[:, col]
            y_col2 = predictions.loc[:, col2]
            kappas[col][col2] = cohen_kappa_score(y_col, y_col2)
            # Positive
            y_col = predictions[predictions['label'] == 1].loc[:, col]
            y_col2 = predictions[predictions['label'] == 1].loc[:, col2]
            kappas_positive[col][col2] = cohen_kappa_score(y_col, y_col2)
            # Negative
            y_col = predictions[predictions['label'] == 0].loc[:, col]
            y_col2 = predictions[predictions['label'] == 0].loc[:, col2]
            kappas_negative[col][col2] = cohen_kappa_score(y_col, y_col2)
    kappas = pd.DataFrame(kappas)
    kappas_positive = pd.DataFrame(kappas_positive)
    kappas_negative = pd.DataFrame(kappas_negative)
    return kappas, kappas_positive, kappas_negative


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    problem = 'mortality'
    dataset_path = os.path.join(parameters['multiclassification_base_path'], parameters[problem + '_directory'],
                                parameters[problem + '_dataset_csv'])

    data_csv = pd.read_csv(dataset_path)
    print("Removing no text constant")
    if parameters['remove_no_text_constant']:
        data_csv = remove_empty_textual_data_episodes(data_csv, 'textual_path')
    print(data_csv['label'].value_counts())

    training_base_directory = parameters['multiclassification_base_path'] + parameters['training_directory_path']
    training_directory = training_base_directory + parameters[problem + "_directory"] \
                         + parameters['execution_saving_path']
    checkpoint_directory = training_directory + parameters['training_checkpoint']
    structured_predictions = None
    textual_predictions = None
    for fold in range(5):
        fold_structured_predictions_path = checkpoint_directory + parameters['fold_structured_predictions_filename'] \
            .format(fold)
        fold_textual_predictions_path = checkpoint_directory + parameters['fold_textual_predictions_filename'] \
            .format(fold)
        fold_structured_predictions = pd.read_csv(fold_structured_predictions_path, index_col=0)
        fold_textual_predictions = pd.read_csv(fold_textual_predictions_path, index_col=0)

        if structured_predictions is None:
            structured_predictions = fold_structured_predictions
            textual_predictions = fold_textual_predictions
        else:
            structured_predictions = structured_predictions.append(fold_structured_predictions)
            textual_predictions = textual_predictions.append(fold_textual_predictions)

    all_predictions = pd.merge(structured_predictions, textual_predictions, left_index=True, right_index=True,
                               how="left")

    kappas, kappas_positive, kappas_negative = kappa_aggreement(all_predictions, data_csv)
    kappas.to_csv(os.path.join(checkpoint_directory, 'kappas.csv'))
    kappas_positive.to_csv(os.path.join(checkpoint_directory, 'kappas_positive.csv'))
    kappas_negative.to_csv(os.path.join(checkpoint_directory, 'kappas_negative.csv'))
