import pickle
import pandas as pd
import math
from matplotlib.ticker import NullFormatter

from multiclassification.results_analysis import calibration_curve
from multiclassification.results_analysis.calibration_curve import plot_model_calibrations
from multiclassification.results_analysis.extract_results_metrics import extract_metrics
from multiclassification.results_analysis.features_missing_values import analyse_missing_values
from multiclassification.results_analysis.longitudinal_mean_plot_curve import get_means, get_mean_for_tableone
from resources.functions import print_with_time, remove_empty_textual_data_episodes
import os
from multiclassification.parameters.classification_parameters import ensemble_parameters as parameters
import matplotlib.pyplot as plt

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

# Loading csv
print_with_time("Loading data")
dataset_path = os.path.join(parameters['multiclassification_base_path'], parameters[problem + '_directory'],
                            parameters[problem + '_dataset_csv'])

data_csv = pd.read_csv(dataset_path)
data_csv = data_csv.sort_values(['episode'])
# data_csv.loc[:, 'structured_path'] = data_csv['structured_path'].apply(lambda x : os.path.join(parameters['multiclassification_base_path'], x))
# data_csv.loc[:, 'textual_path'] = data_csv['textual_path'].apply(lambda x : os.path.join(parameters['multiclassification_base_path'], x))

if parameters['remove_no_text_constant']:
    if os.path.exists(os.path.join(analysis_directory, "data.csv")):
        data_csv = pd.read_csv(os.path.join(analysis_directory, "data.csv"))
    else:
        data_csv = remove_empty_textual_data_episodes(data_csv, 'textual_path')
        data_csv.to_csv(os.path.join(analysis_directory, "data.csv"))
print(data_csv['label'].value_counts())
positive_df = data_csv[data_csv['label'] == 1]
negative_df = data_csv[data_csv['label'] == 0]

metrics = ["w_fscore", "w_precision", "w_recall", "ma_fscore", "recall_b", "auc"]
columns_order = ["Model", "Data Type", "Precision", "Recall", "W F-Score", "U F-Score", "TP Rate", "AUC"]

level0_results_path = os.path.join(checkpoint_directory, parameters['metrics_filename'])
ensemble_results_path = os.path.join(checkpoint_directory, "ensemble_result.csv")
structured_results_path = os.path.join(checkpoint_directory, 'metadata_results.csv')
evaluation_balanced_dataset_results_path = None
if parameters['balance_training_data']:
    evaluation_balanced_dataset_results_path = os.path.join(checkpoint_directory, parameters['fold_evaluation_metrics_filename'])
print("Extracting metrics")
extract_metrics(analysis_directory, level0_results_path, structured_results_path, ensemble_results_path,
                balanced_test_set_results=evaluation_balanced_dataset_results_path)

if not os.path.exists(os.path.join(analysis_directory, "variables_mean.csv")):
    means = get_mean_for_tableone(data_csv, 'structured_path')
    means.to_csv(os.path.join(analysis_directory, "variables_mean.csv"))

# Calibration curve
# Ensemble
metamodel_predictions_path = os.path.join(checkpoint_directory, 'metamodels_predictions.csv')
metamodel_predictions_df = pd.read_csv(metamodel_predictions_path)
models_name = metamodel_predictions_df["model"].unique().tolist()
data_origins = metamodel_predictions_df["origin"].unique().tolist()
print(models_name)
probabilites_model_series = dict()
print("Calculating calibration curves")
for model in models_name:
    if model == "DecisionTreeClassifier":
        continue
    model_predictions = metamodel_predictions_df[metamodel_predictions_df["model"] == model]
    for origin in data_origins:
        model_origin_predictions = model_predictions[model_predictions["origin"] == origin]
        model_origin_predictions = model_origin_predictions.set_index(["episode"])
        model_probabilities = model_origin_predictions["probas"]
        model_probabilities.name = model
        if origin not in probabilites_model_series.keys():
            probabilites_model_series[origin] = model_probabilities
        else:
            probabilites_model_series[origin] = pd.concat([probabilites_model_series[origin], model_probabilities], axis=1)

calibration_directory = os.path.join(analysis_directory, "calibration")
if not os.path.exists(calibration_directory):
    os.makedirs(calibration_directory)
# Merge label
for key in probabilites_model_series.keys():
    probabilites_model_series[key] = pd.merge(probabilites_model_series[key], data_csv[["episode", "label"]],
                                              left_on="episode", right_on="episode")
    plot_model_calibrations(probabilites_model_series[key], "label", calibration_directory, key)

# Level-0
structured_probabilities = pd.read_csv(os.path.join(checkpoint_directory, parameters['structured_predictions_filename']),
                                       index_col=0)
structured_probabilities = pd.merge(structured_probabilities, data_csv[["episode", "label"]],
                                              left_index=True, right_on="episode")
plot_model_calibrations(structured_probabilities, "label", calibration_directory, "sts")

textual_probabilities = pd.read_csv(os.path.join(checkpoint_directory, parameters['textual_predictions_filename']),
                                    index_col=0)
textual_probabilities = pd.merge(textual_probabilities, data_csv[["episode", "label"]],
                                              left_index=True, right_on="episode")
plot_model_calibrations(textual_probabilities, "label", calibration_directory, "tts")

# Missing values
missing_values_path = os.path.join(analysis_directory, "missing")
if not os.path.exists(missing_values_path):
    os.makedirs(missing_values_path)
total_missing_values_path = os.path.join(missing_values_path, "total")
if not os.path.exists(total_missing_values_path):
    os.makedirs(total_missing_values_path)
if not os.path.exists(os.path.join(total_missing_values_path, "total.csv")) and \
    not os.path.exists(os.path.join(total_missing_values_path, "events_time.csv")):
    total_events, total_events_time = analyse_missing_values(data_csv, "structured_path")
    total_events.to_csv(os.path.join(total_missing_values_path, "total.csv"))
    total_events_time.to_csv(os.path.join(total_missing_values_path, "events_time.csv"))

positive_missing_values_path = os.path.join(missing_values_path, "positive")
if not os.path.exists(positive_missing_values_path):
    os.makedirs(positive_missing_values_path)
if not os.path.exists(os.path.join(positive_missing_values_path, "total.csv")) and \
    not os.path.exists(os.path.join(positive_missing_values_path, "events_time.csv")):
    total_events, total_events_time = analyse_missing_values(positive_df, "structured_path")
    total_events.to_csv(os.path.join(positive_missing_values_path, "total.csv"))
    total_events_time.to_csv(os.path.join(positive_missing_values_path, "events_time.csv"))

negative_missing_values_path = os.path.join(missing_values_path, "negative")
if not os.path.exists(negative_missing_values_path):
    os.makedirs(negative_missing_values_path)
if not os.path.exists(os.path.join(negative_missing_values_path, "total.csv")) and \
    not os.path.exists(os.path.join(negative_missing_values_path, "events_time.csv")):
    total_events, total_events_time = analyse_missing_values(negative_df, "structured_path")
    total_events.to_csv(os.path.join(negative_missing_values_path, "total.csv"))
    total_events_time.to_csv(os.path.join(negative_missing_values_path, "events_time.csv"))

# Ploting attributes mean
variables_directory = os.path.join(analysis_directory, "variables")
if not os.path.exists(variables_directory):
    os.makedirs(variables_directory)

# Positive vs negative
positive_vs_negative_features_path = os.path.join(variables_directory, "positive_negative")
if not os.path.exists(positive_vs_negative_features_path):
    os.makedirs(positive_vs_negative_features_path)
if not os.path.exists(os.path.join(positive_vs_negative_features_path, "positive_vs_negative.jpg")):
    if not os.path.exists(os.path.join(positive_vs_negative_features_path, "positive_mean_values.csv")):
        mean_positive_df, raw, num_values = get_means(positive_df, 'structured_path')
        raw.to_csv(os.path.join(positive_vs_negative_features_path, "positive_raw_values.csv"))
        num_values = pd.DataFrame(num_values)
        num_values.to_csv(os.path.join(positive_vs_negative_features_path, "positive_num_values.csv"))
        mean_positive_df.to_csv(os.path.join(positive_vs_negative_features_path, "positive_mean_values.csv"))
    else:
        mean_positive_df = pd.read_csv(os.path.join(positive_vs_negative_features_path, "positive_mean_values.csv"))

    if not os.path.exists(os.path.join(positive_vs_negative_features_path, "negative_mean_values.csv")):
        mean_negative_df, raw, num_values = get_means(negative_df, 'structured_path')
        raw.to_csv(os.path.join(positive_vs_negative_features_path, "negative_raw_values.csv"))
        num_values = pd.DataFrame(num_values)
        num_values.to_csv(os.path.join(positive_vs_negative_features_path, "negative_num_values.csv"))
        mean_negative_df.to_csv(os.path.join(positive_vs_negative_features_path, "negative_mean_values.csv"))
    else:
        mean_negative_df = pd.read_csv(os.path.join(positive_vs_negative_features_path, "negative_mean_values.csv"))

    features = [column for column in mean_positive_df.columns if not "bucket" in column and column != "starttime"
                    and column != "endtime" and '_min' not in column and '_max' not in column]
    print("Features {}".format(len(features)))
    mean_positive_df = mean_positive_df.sort_values(by=["bucket"])
    mean_negative_df = mean_negative_df.sort_values(by=["bucket"])
    for feature in features:
        fig = plt.figure(0, figsize=(10, 5))
        # ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        buckets = mean_positive_df['bucket'].values
        positive_values = mean_positive_df[feature].values
        negative_values = mean_negative_df[feature].values
        plt.plot(buckets, positive_values, label="Positive")
        plt.plot(buckets, negative_values, label="Negative")
        plt.xlim([0, 48])
        plt.legend(loc="lower right")
        plt.title(feature.replace("_", " ").title())
        plt.xlabel("Time in ICU (H)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(positive_vs_negative_features_path, "{}_positive_negative.jpg".format(feature)))
        plt.clf()

    # subplot_cols = 3
    # subplot_rows = int(math.ceil(len(features) / subplot_cols))
    # subplot_id = 0
    # column_count = 0
    # fig, subplots = plt.subplots(subplot_rows, subplot_cols, figsize=(100, 100))
    # for subplot_row in range(subplot_rows):
    #     if column_count == len(features):
    #         break
    #     for subplot_col in range(subplot_cols):
    #         if column_count == len(features):
    #             break
    #         column = features[column_count]
    #         subplot = subplots[subplot_row, subplot_col]
    #         buckets = mean_positive_df['bucket'].values
    #         positive_values = mean_positive_df[column].values
    #         negative_values = mean_negative_df[column].values
    #         subplot.plot(buckets, positive_values, label="Positive")
    #         subplot.plot(buckets, negative_values, label="Negative")
    #         subplot.set_xlim([0, 48])
    #         subplot.legend(loc="lower right")
    #         subplot.set_title(column.replace("_", " ").title())
    #         subplot.grid(True)
    #         column_count += 1
    # # Adjust the subplot layout, because the logit one may take more space
    # # than usual, due to y-tick labels like "1 - 10^{-3}"
    # fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.75,
    #                     wspace=0.35)
    # plt.savefig(os.path.join(variables_directory, "positive_vs_negative.jpg"))
# For each ensemble, do the same TP vs FN
probabilites_model_series = dict()
for model in models_name:
    model_predictions = metamodel_predictions_df[metamodel_predictions_df["model"] == model]
    for origin in data_origins:
        model_origin_predictions = model_predictions[model_predictions["origin"] == origin]
        model_origin_predictions = model_origin_predictions.set_index(["episode"], drop=False)
        model_probabilities = model_origin_predictions["classes"]
        model_probabilities.name = model
        if origin not in probabilites_model_series.keys():
            probabilites_model_series[origin] = model_probabilities
        else:
            probabilites_model_series[origin] = pd.concat([probabilites_model_series[origin], model_probabilities], axis=1)

for origin in probabilites_model_series.keys():
    probabilites_model_series[origin] = pd.merge(probabilites_model_series[origin], data_csv[["episode", "label"]],
                                                 left_on="episode", right_on="episode")
    origin_variables_tp_vs_fn_path = os.path.join(variables_directory, "{}_tp_vs_fn".format(origin))
    if not os.path.exists(origin_variables_tp_vs_fn_path):
        os.makedirs(origin_variables_tp_vs_fn_path)
    for column in probabilites_model_series[origin].columns:
        if 'Unnamed' in column or 'episode' in column or column == 'label':
            continue
        model_variables_tp_vs_fn_path = os.path.join(origin_variables_tp_vs_fn_path, column)
        if not os.path.exists(model_variables_tp_vs_fn_path):
            os.makedirs(model_variables_tp_vs_fn_path)
        classes_df = probabilites_model_series[origin][['episode', column, 'label']]

        if not os.path.exists(os.path.join(model_variables_tp_vs_fn_path, "{}_tp.csv".format(column))):
            tp_df = classes_df[(classes_df['label'] == 1) & (classes_df[column] == 1)]
            tp_df = data_csv[data_csv['episode'].isin(tp_df['episode'].tolist())]
            tp_df.to_csv(os.path.join(model_variables_tp_vs_fn_path, "{}_tp.csv".format(column)))
        else:
            tp_df = pd.read_csv(os.path.join(model_variables_tp_vs_fn_path, "{}_tp.csv".format(column)))
            tp_df = tp_df.loc[:, ~tp_df.columns.str.match("Unnamed")]

        if not os.path.exists(os.path.join(model_variables_tp_vs_fn_path, "{}_fn.csv".format(column))):
            fn_df = classes_df[(classes_df['label'] == 1) & (classes_df[column] == 0)]
            fn_df = data_csv[data_csv['episode'].isin(fn_df['episode'].tolist())]
            fn_df.to_csv(os.path.join(model_variables_tp_vs_fn_path, "{}_fn.csv".format(column)))
        else:
            fn_df = pd.read_csv(os.path.join(model_variables_tp_vs_fn_path, "{}_fn.csv".format(column)))
            fn_df = fn_df.loc[:, ~fn_df.columns.str.match("Unnamed")]

        if not os.path.exists(os.path.join(model_variables_tp_vs_fn_path, "tp_{}_mean_values.csv".format(column))):
            mean_tp_df, raw, num_values = get_means(tp_df, 'structured_path')
            raw.to_csv(os.path.join(model_variables_tp_vs_fn_path, "tp_{}_raw_values.csv".format(column)))
            num_values = pd.DataFrame(num_values)
            num_values.to_csv(os.path.join(model_variables_tp_vs_fn_path, "tp_{}_num_values.csv".format(column)))
            mean_tp_df.to_csv(os.path.join(model_variables_tp_vs_fn_path, "tp_{}_mean_values.csv".format(column)))
        else:
            mean_tp_df = pd.read_csv(os.path.join(model_variables_tp_vs_fn_path, "tp_{}_mean_values.csv".format(column)))
            mean_tp_df = mean_tp_df.loc[:, ~mean_tp_df.columns.str.match("Unnamed")]

        if not os.path.exists(os.path.join(model_variables_tp_vs_fn_path, "fn_{}_mean_values.csv".format(column))):
            mean_fn_df, raw, num_values = get_means(fn_df, 'structured_path')
            raw.to_csv(os.path.join(model_variables_tp_vs_fn_path, "fn_{}_raw_values.csv".format(column)))
            num_values = pd.DataFrame(num_values)
            num_values.to_csv(os.path.join(model_variables_tp_vs_fn_path, "fn_{}_num_values.csv".format(column)))
            mean_fn_df.to_csv(os.path.join(model_variables_tp_vs_fn_path, "fn_{}_mean_values.csv".format(column)))
        else:
            mean_fn_df = pd.read_csv(os.path.join(model_variables_tp_vs_fn_path, "fn_{}_mean_values.csv".format(column)))
            mean_fn_df= mean_fn_df.loc[:, ~mean_fn_df.columns.str.match("Unnamed")]

        for feature in features:
            if 'Unnamed' in feature:
                continue
            fig = plt.figure(0, figsize=(10, 5))
            buckets = mean_tp_df['bucket'].values
            tp_values = mean_tp_df[feature].values
            plt.plot(buckets, tp_values, label="TP")
            buckets = mean_fn_df["bucket"].values
            fn_values = mean_fn_df[feature].values
            plt.plot(buckets, fn_values, label="FN")
            plt.xlim([0, min(len(tp_values), len(fn_values))])
            plt.legend(loc="lower right")
            plt.title(feature.replace("_", " ").title())
            plt.xlabel("Time in ICU (H)")
            plt.ylabel("Value")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(model_variables_tp_vs_fn_path, "{}.jpg".format(feature)))
            plt.clf()
